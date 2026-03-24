"""
OpenRCA Verifiers Environment

Implements the OpenRCA benchmark (ICLR 2025) as a Verifiers environment for
evaluating and training LLMs on root cause analysis of software failures.

The environment provides Python code execution tools for analyzing telemetry
data (metrics, traces, logs) from microservice systems to identify root causes
of failures.

Reference: https://github.com/microsoft/OpenRCA
Paper: https://openreview.net/forum?id=M4qNIzQYpd
"""

import ast
import asyncio
import itertools
import json
import os
import re
import sys
import threading
from datetime import datetime
from io import StringIO

import verifiers as vf
from datasets import Dataset


# ─── Constants ───────────────────────────────────────────────────────────

MAX_OUTPUT_CHARS = 50000
EXEC_TIMEOUT_SECONDS = 60

ALL_SYSTEMS = ["Bank", "Market/cloudbed-1", "Market/cloudbed-2", "Telecom"]


# ─── System prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a DevOps assistant specialized in root cause analysis (RCA) of "
    "software failures in microservice systems. You analyze telemetry data "
    "(metrics, traces, and logs) to identify the root causes of system failures.\n"
    "\n"
    "## Diagnosis Methodology\n"
    "\n"
    "Follow this workflow for failure diagnosis:\n"
    "\n"
    "1. **Preprocess**: Aggregate each KPI of each component to obtain multiple "
    "time series classified by 'component-KPI'. Calculate global thresholds "
    "(e.g., P95) using the ENTIRE KPI series within a metric file BEFORE "
    "filtering to the failure time window.\n"
    "2. **Anomaly Detection**: Identify data points exceeding global thresholds. "
    "Look for drops in traffic/business KPIs (e.g., success rate) using lower "
    "thresholds (<=P5). Loosen thresholds (e.g., P95->P90) if no anomalies found.\n"
    "3. **Fault Identification**: Find consecutive anomaly sub-series per "
    "component-KPI. Filter out isolated noise spikes. Exclude borderline "
    "threshold breaches (<50% excess over threshold) as likely false positives.\n"
    "4. **Root Cause Localization**: Determine which fault is the root cause:\n"
    "   - If faults at different levels, pick the level with most significant "
    "deviation (>>50% over threshold).\n"
    "   - If multiple faulty services, the root cause is the most downstream "
    "FAULTY service in the trace call chain.\n"
    "   - If multiple faulty containers, same rule applies at container level.\n"
    "   - Node-level faults do not propagate through traces.\n"
    "\n"
    "Key rules:\n"
    "- Always calculate global thresholds BEFORE filtering to the failure window.\n"
    "- Analysis order: threshold calculation -> data extraction -> metrics -> "
    "traces -> logs.\n"
    "- Use metrics first to narrow the search space, then traces, then logs.\n"
    "- Do not focus solely on error logs; info logs contain critical information.\n"
    "- Do not visualize data. Provide only text-based analysis.\n"
    "- Use UTC+8 timezone (pytz.timezone('Asia/Shanghai')) for all analysis.\n"
    "\n"
    "## Tools\n"
    "\n"
    "You have access to these tools:\n"
    "- `execute_python`: Run Python code in a persistent environment with pandas, "
    "numpy, and other libraries pre-imported. The variable `DATA_DIR` contains "
    "the path to the telemetry data root directory.\n"
    "- `list_directory`: List files and directories within the telemetry data "
    "directory.\n"
    "\n"
    "## Output Format\n"
    "\n"
    "When you have completed your analysis, provide your final answer as a JSON "
    "object. Only include the fields requested by the task:\n"
    "\n"
    "```json\n"
    "{\n"
    '    "1": {\n'
    '        "root cause occurrence datetime": "[%Y-%m-%d %H:%M:%S]",\n'
    '        "root cause component": "[COMPONENT]",\n'
    '        "root cause reason": "[REASON]"\n'
    "    }\n"
    "}\n"
    "```\n"
    "\n"
    "If there are multiple failures, number them chronologically (1, 2, 3, ...). "
    "Select components and reasons ONLY from the provided candidate lists. "
    "Do not reply 'unknown' or 'null' or 'not found'. Be decisive and infer "
    "a possible answer based on your observations."
)


# ─── Per-system telemetry schemas ────────────────────────────────────────

BANK_SCHEMA = (
    "## TELEMETRY DIRECTORY STRUCTURE\n"
    "\n"
    "- Telemetry directory: `os.path.join(DATA_DIR, 'Bank/telemetry/')`\n"
    "- Subdirectories organized by date, e.g., `Bank/telemetry/2021_03_05/`\n"
    "- Within each date directory: `metric/`, `trace/`, and `log/` subdirectories\n"
    "- Data is stored in CSV format\n"
    "\n"
    "## DATA SCHEMA\n"
    "\n"
    "1. **Metric Files**:\n"
    "\n"
    "    1. `metric_app.csv`:\n"
    "        ```csv\n"
    "        timestamp,rr,sr,cnt,mrt,tc\n"
    "        1614787440,100.0,100.0,22,53.27,ServiceTest1\n"
    "        ```\n"
    "\n"
    "    2. `metric_container.csv`:\n"
    "        ```csv\n"
    "        timestamp,cmdb_id,kpi_name,value\n"
    "        1614787200,Tomcat04,OSLinux-CPU_CPU_CPUCpuUtil,26.2957\n"
    "        ```\n"
    "\n"
    "2. **Trace Files**:\n"
    "\n"
    "    1. `trace_span.csv`:\n"
    "        ```csv\n"
    "        timestamp,cmdb_id,parent_id,span_id,trace_id,duration\n"
    "        1614787199628,dockerA2,369-bcou-dle-way1-c514cf30-43410@0824-"
    "2f0e47a816-17492,21030300016145905763,gw0120210304000517192504,19\n"
    "        ```\n"
    "\n"
    "3. **Log Files**:\n"
    "\n"
    "    1. `log_service.csv`:\n"
    "        ```csv\n"
    "        log_id,timestamp,cmdb_id,log_name,value\n"
    "        8c7f5908ed126abdd0de6dbdd739715c,1614787201,Tomcat01,gc,"
    '"3748789.580: [GC (CMS Initial Mark) ...]"\n'
    "        ```\n"
    "\n"
    "## CLARIFICATION\n"
    "\n"
    "1. This microservice system is a banking platform.\n"
    "2. `metric_app.csv` contains four KPIs: rr, sr, cnt, mrt. "
    "`metric_container.csv` records various KPIs (CPU, memory, etc.) "
    "in the `kpi_name` field.\n"
    "3. Timestamp units: Metric = seconds (e.g., 1614787440), "
    "Trace = milliseconds (e.g., 1614787199628), "
    "Log = seconds (e.g., 1614787201).\n"
    "4. Use UTC+8 timezone (`pytz.timezone('Asia/Shanghai')`) for all analysis."
)

BANK_CANDIDATES = (
    "## POSSIBLE ROOT CAUSE REASONS\n"
    "\n"
    "- high CPU usage\n"
    "- high memory usage\n"
    "- network latency\n"
    "- network packet loss\n"
    "- high disk I/O read usage\n"
    "- high disk space usage\n"
    "- high JVM CPU load\n"
    "- JVM Out of Memory (OOM) Heap\n"
    "\n"
    "## POSSIBLE ROOT CAUSE COMPONENTS\n"
    "\n"
    "- apache01\n"
    "- apache02\n"
    "- Tomcat01\n"
    "- Tomcat02\n"
    "- Tomcat04\n"
    "- Tomcat03\n"
    "- MG01\n"
    "- MG02\n"
    "- IG01\n"
    "- IG02\n"
    "- Mysql01\n"
    "- Mysql02\n"
    "- Redis01\n"
    "- Redis02"
)

MARKET_SCHEMA = (
    "## TELEMETRY DIRECTORY STRUCTURE\n"
    "\n"
    "- Telemetry directories for two cloudbeds:\n"
    "  `os.path.join(DATA_DIR, 'Market/cloudbed-1/telemetry/')` and\n"
    "  `os.path.join(DATA_DIR, 'Market/cloudbed-2/telemetry/')`\n"
    "- Subdirectories organized by date, e.g., "
    "`Market/cloudbed-1/telemetry/2022_03_20/`\n"
    "- Within each date directory: `metric/`, `trace/`, and `log/` "
    "subdirectories\n"
    "- Data is stored in CSV format\n"
    "\n"
    "## DATA SCHEMA\n"
    "\n"
    "1. **Metric Files**:\n"
    "\n"
    "    1. `metric_container.csv`:\n"
    "        ```csv\n"
    "        timestamp,cmdb_id,kpi_name,value\n"
    "        1647781200,node-6.adservice2-0,"
    "container_fs_writes_MB./dev/vda,0.0\n"
    "        ```\n"
    "\n"
    "    2. `metric_mesh.csv`:\n"
    "        ```csv\n"
    "        timestamp,cmdb_id,kpi_name,value\n"
    "        1647790380,cartservice-1.source.cartservice.redis-cart,"
    "istio_tcp_sent_bytes.-,1255.0\n"
    "        ```\n"
    "\n"
    "    3. `metric_node.csv`:\n"
    "        ```csv\n"
    "        timestamp,cmdb_id,kpi_name,value\n"
    "        1647705600,node-1,system.cpu.iowait,0.31\n"
    "        ```\n"
    "\n"
    "    4. `metric_runtime.csv`:\n"
    "        ```csv\n"
    "        timestamp,cmdb_id,kpi_name,value\n"
    "        1647730800,adservice.ts:8088,"
    "java_nio_BufferPool_TotalCapacity.direct,57343.0\n"
    "        ```\n"
    "\n"
    "    5. `metric_service.csv`:\n"
    "        ```csv\n"
    "        service,timestamp,rr,sr,mrt,count\n"
    "        adservice-grpc,1647716400,100.0,100.0,2.429508196728182,61\n"
    "        ```\n"
    "\n"
    "2. **Trace Files**:\n"
    "\n"
    "    1. `trace_span.csv`:\n"
    "        ```csv\n"
    "        timestamp,cmdb_id,span_id,trace_id,duration,type,status_code,"
    "operation_name,parent_span\n"
    "        1647705600361,frontend-0,a652d4d10e9478fc,"
    "9451fd8fdf746a80687451dae4c4e984,49877,rpc,0,"
    "hipstershop.CheckoutService/PlaceOrder,952754a738a11675\n"
    "        ```\n"
    "\n"
    "3. **Log Files**:\n"
    "\n"
    "    1. `log_proxy.csv`:\n"
    "        ```csv\n"
    "        log_id,timestamp,cmdb_id,log_name,value\n"
    "        KN43pn8BmS57GQLkQUdP,1647761110,cartservice-1,"
    "log_cartservice-service_application,"
    "etCartAsync called with userId=...\n"
    "        ```\n"
    "\n"
    "    2. `log_service.csv`:\n"
    "        ```csv\n"
    "        log_id,timestamp,cmdb_id,log_name,value\n"
    "        GIvpon8BDiVcQfZwJ5a9,1647705660,currencyservice-0,"
    "log_currencyservice-service_application,"
    '"severity: info, message: Getting supported currencies..."\n'
    "        ```\n"
    "\n"
    "## CLARIFICATION\n"
    "\n"
    "1. This microservice system is an E-commerce platform with a failover "
    "mechanism, with each service deployed across four pods. A container (pod) "
    "can be deployed on different nodes. If the root cause is a single pod, the "
    "failure may not significantly impact service metrics. If ALL pods of a "
    "service are faulty, service metrics will be significantly impacted and "
    "faults can propagate through the call chain.\n"
    "2. `metric_service.csv` contains four KPIs: rr, sr, mrt, count. Other "
    "metric files record various KPIs in the `kpi_name` field.\n"
    "3. `cmdb_id` naming conventions:\n"
    "   - Container metrics: `<node>-x.<service>-x` (e.g., `node-1.adservice-0`)\n"
    "   - Node metrics: `<node>-x` (e.g., `node-1`)\n"
    "   - Service metrics: `<service>-grpc` (e.g., `adservice-grpc`)\n"
    "   - Traces and Logs: `<service>-x` (e.g., `frontend-0`)\n"
    "4. Timestamp units: Metrics = seconds, Traces = milliseconds, "
    "Logs = seconds.\n"
    "5. Pod equals Container in this system.\n"
    "6. Use UTC+8 timezone (`pytz.timezone('Asia/Shanghai')`) for all analysis."
)

MARKET_CANDIDATES = (
    "## POSSIBLE ROOT CAUSE COMPONENTS\n"
    "\n"
    "(if the root cause is at the node level)\n"
    "- node-1\n"
    "- node-2\n"
    "- node-3\n"
    "- node-4\n"
    "- node-5\n"
    "- node-6\n"
    "\n"
    "(if the root cause is at the pod level)\n"
    "- frontend-0\n"
    "- frontend-1\n"
    "- frontend-2\n"
    "- frontend2-0\n"
    "- shippingservice-0\n"
    "- shippingservice-1\n"
    "- shippingservice-2\n"
    "- shippingservice2-0\n"
    "- checkoutservice-0\n"
    "- checkoutservice-1\n"
    "- checkoutservice-2\n"
    "- checkoutservice2-0\n"
    "- currencyservice-0\n"
    "- currencyservice-1\n"
    "- currencyservice-2\n"
    "- currencyservice2-0\n"
    "- adservice-0\n"
    "- adservice-1\n"
    "- adservice-2\n"
    "- adservice2-0\n"
    "- emailservice-0\n"
    "- emailservice-1\n"
    "- emailservice-2\n"
    "- emailservice2-0\n"
    "- cartservice-0\n"
    "- cartservice-1\n"
    "- cartservice-2\n"
    "- cartservice2-0\n"
    "- productcatalogservice-0\n"
    "- productcatalogservice-1\n"
    "- productcatalogservice-2\n"
    "- productcatalogservice2-0\n"
    "- recommendationservice-0\n"
    "- recommendationservice-1\n"
    "- recommendationservice-2\n"
    "- recommendationservice2-0\n"
    "- paymentservice-0\n"
    "- paymentservice-1\n"
    "- paymentservice-2\n"
    "- paymentservice2-0\n"
    "\n"
    "(if the root cause is at the service level)\n"
    "- frontend\n"
    "- shippingservice\n"
    "- checkoutservice\n"
    "- currencyservice\n"
    "- adservice\n"
    "- emailservice\n"
    "- cartservice\n"
    "- productcatalogservice\n"
    "- recommendationservice\n"
    "- paymentservice\n"
    "\n"
    "## POSSIBLE ROOT CAUSE REASONS\n"
    "\n"
    "- container CPU load\n"
    "- container memory load\n"
    "- container network packet retransmission\n"
    "- container network packet corruption\n"
    "- container network latency\n"
    "- container packet loss\n"
    "- container process termination\n"
    "- container read I/O load\n"
    "- container write I/O load\n"
    "- node CPU load\n"
    "- node CPU spike\n"
    "- node memory consumption\n"
    "- node disk read I/O consumption\n"
    "- node disk write I/O consumption\n"
    "- node disk space consumption"
)

TELECOM_SCHEMA = (
    "## TELEMETRY DIRECTORY STRUCTURE\n"
    "\n"
    "- Telemetry directory: `os.path.join(DATA_DIR, 'Telecom/telemetry/')`\n"
    "- Subdirectories organized by date, e.g., `Telecom/telemetry/2020_04_11/`\n"
    "- Within each date directory: `metric/` and `trace/` subdirectories "
    "(no log directory for this system)\n"
    "- Data is stored in CSV format\n"
    "\n"
    "## DATA SCHEMA\n"
    "\n"
    "1. **Metric Files**:\n"
    "\n"
    "    1. `metric_app.csv`:\n"
    "        ```csv\n"
    "        serviceName,startTime,avg_time,num,succee_num,succee_rate\n"
    "        osb_001,1586534400000,0.333,1,1,1.0\n"
    "        ```\n"
    "\n"
    "    2. `metric_container.csv`:\n"
    "        ```csv\n"
    "        itemid,name,bomc_id,timestamp,value,cmdb_id\n"
    "        999999996381330,container_mem_used,ZJ-004-060,"
    "1586534423000,59.000000,docker_008\n"
    "        ```\n"
    "\n"
    "    3. `metric_middleware.csv`:\n"
    "        ```csv\n"
    "        itemid,name,bomc_id,timestamp,value,cmdb_id\n"
    "        999999996508323,connected_clients,ZJ-005-024,"
    "1586534672000,25,redis_003\n"
    "        ```\n"
    "\n"
    "    4. `metric_node.csv`:\n"
    "        ```csv\n"
    "        itemid,name,bomc_id,timestamp,value,cmdb_id\n"
    "        999999996487783,CPU_iowait_time,ZJ-001-010,"
    "1586534683000,0.022954,os_017\n"
    "        ```\n"
    "\n"
    "    5. `metric_service.csv`:\n"
    "        ```csv\n"
    "        itemid,name,bomc_id,timestamp,value,cmdb_id\n"
    "        999999998650974,MEM_Total,ZJ-002-055,"
    "1586534694000,381.902264,db_003\n"
    "        ```\n"
    "\n"
    "2. **Trace Files**:\n"
    "\n"
    "    1. `trace_span.csv`:\n"
    "        ```csv\n"
    "        callType,startTime,elapsedTime,success,traceId,id,pid,"
    "cmdb_id,dsName,serviceName\n"
    "        JDBC,1586534400335,2.0,True,01df517164d1c0365586,"
    "407d617164d1c14f2613,6e02217164d1c14b2607,docker_006,db_003,\n"
    "        ```\n"
    "\n"
    "## CLARIFICATION\n"
    "\n"
    "1. This service system is a telecom database system.\n"
    "2. `metric_app.csv` contains five KPIs: startTime, avg_time, num, "
    "succee_num, succee_rate. Other metric files record various KPIs "
    "in the `name` field.\n"
    "3. All telemetry timestamps are in milliseconds.\n"
    "4. Use UTC+8 timezone (`pytz.timezone('Asia/Shanghai')`) for all analysis."
)

TELECOM_CANDIDATES = (
    "## POSSIBLE ROOT CAUSE REASONS\n"
    "\n"
    "- CPU fault\n"
    "- network delay\n"
    "- network loss\n"
    "- db connection limit\n"
    "- db close\n"
    "\n"
    "## POSSIBLE ROOT CAUSE COMPONENTS\n"
    "\n"
    "(if the root cause is at the node level)\n"
    "- os_001\n"
    "- os_002\n"
    "- os_003\n"
    "- os_004\n"
    "- os_005\n"
    "- os_006\n"
    "- os_007\n"
    "- os_008\n"
    "- os_009\n"
    "- os_010\n"
    "- os_011\n"
    "- os_012\n"
    "- os_013\n"
    "- os_014\n"
    "- os_015\n"
    "- os_016\n"
    "- os_017\n"
    "- os_018\n"
    "- os_019\n"
    "- os_020\n"
    "- os_021\n"
    "- os_022\n"
    "\n"
    "(if the root cause is at the pod level)\n"
    "- docker_001\n"
    "- docker_002\n"
    "- docker_003\n"
    "- docker_004\n"
    "- docker_005\n"
    "- docker_006\n"
    "- docker_007\n"
    "- docker_008\n"
    "\n"
    "(if the root cause is at the service level)\n"
    "- db_001\n"
    "- db_002\n"
    "- db_003\n"
    "- db_004\n"
    "- db_005\n"
    "- db_006\n"
    "- db_007\n"
    "- db_008\n"
    "- db_009\n"
    "- db_010\n"
    "- db_011\n"
    "- db_012\n"
    "- db_013"
)

SYSTEM_INFO: dict[str, dict[str, str]] = {
    "Bank": {"schema": BANK_SCHEMA, "candidates": BANK_CANDIDATES},
    "Market/cloudbed-1": {"schema": MARKET_SCHEMA, "candidates": MARKET_CANDIDATES},
    "Market/cloudbed-2": {"schema": MARKET_SCHEMA, "candidates": MARKET_CANDIDATES},
    "Telecom": {"schema": TELECOM_SCHEMA, "candidates": TELECOM_CANDIDATES},
}


# ─── Evaluation logic ───────────────────────────────────────────────────

def _time_within_one_minute(time1_str: str, time2_str: str) -> bool:
    """Check if two timestamp strings are within 1 minute of each other."""
    time_format = "%Y-%m-%d %H:%M:%S"
    try:
        t1 = datetime.strptime(time1_str.strip(), time_format)
        t2 = datetime.strptime(time2_str.strip(), time_format)
        return abs(t1 - t2).total_seconds() <= 60
    except ValueError:
        return False


def evaluate_prediction(
    prediction: str, scoring_points: str
) -> tuple[list[str], list[str], float]:
    """Evaluate a prediction against OpenRCA scoring criteria.

    This is a faithful port of the evaluate() function from the original
    OpenRCA benchmark (main/evaluate.py).

    Args:
        prediction: The model's prediction text containing JSON root cause info.
        scoring_points: The scoring criteria string from the query dataset.

    Returns:
        A tuple of (passed_criteria, failed_criteria, score).
    """
    predict_pattern = (
        r"{\s*"
        r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
        r'(?:"root cause component":\s*"(.*?)")?,?\s*'
        r'(?:"root cause reason":\s*"(.*?)")?\s*}'
    )
    predict_matches = re.findall(predict_pattern, prediction)

    predict_results = []
    for match in predict_matches:
        datetime_str, component, reason = match
        if datetime_str or component or reason:
            predict_results.append(
                {
                    "root cause occurrence datetime": datetime_str,
                    "root cause component": component,
                    "root cause reason": reason,
                }
            )

    prediction_length = len(predict_results)

    component_pattern = (
        r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
    )
    reason_pattern = (
        r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
    )
    time_pattern = (
        r"The (?:\d+-th|only) root cause occurrence time is within "
        r"1 minutes \(i\.e\., <=1min\) of ([^\n]+)"
    )

    components = re.findall(component_pattern, scoring_points)
    reasons = re.findall(reason_pattern, scoring_points)
    times = re.findall(time_pattern, scoring_points)

    scoringpoints_length = max(len(components), len(reasons), len(times), 1)
    scores_num = len(components) + len(reasons) + len(times)

    if scores_num == 0:
        return [], [], 0.0

    scores_get = 0
    passing_criteria: list[str] = []

    if scoringpoints_length == prediction_length:
        best_score = -1
        for perm in itertools.permutations(predict_results):
            current_score = 0
            current_passing: list[str] = []
            for i in range(scoringpoints_length):
                if len(components) == scoringpoints_length:
                    if perm[i]["root cause component"] == components[i]:
                        current_score += 1
                        current_passing.append(components[i])
                if len(reasons) == scoringpoints_length:
                    if perm[i]["root cause reason"] == reasons[i]:
                        current_score += 1
                        current_passing.append(reasons[i])
                if len(times) == scoringpoints_length:
                    if _time_within_one_minute(
                        times[i],
                        perm[i]["root cause occurrence datetime"],
                    ):
                        current_score += 1
                        current_passing.append(times[i])
            if current_score > best_score:
                best_score = current_score
                passing_criteria = current_passing
        scores_get = best_score

    failing_criteria = list(
        set(components + reasons + times) - set(passing_criteria)
    )

    final_score = scores_get / scores_num
    return passing_criteria, failing_criteria, round(final_score, 2)


# ─── Reward functions ────────────────────────────────────────────────────

async def openrca_score(completion: list[dict[str, str]], answer: str) -> float:
    """Score the model's prediction against OpenRCA scoring criteria.

    Extracts the JSON prediction from the model's final response and evaluates
    it against the scoring points from the dataset.
    """
    prediction_text = ""
    for msg in reversed(completion):
        if msg.get("role") == "assistant" and msg.get("content"):
            prediction_text = msg["content"]
            break

    if not prediction_text:
        return 0.0

    _, _, score = evaluate_prediction(prediction_text, answer)
    return score


async def difficulty_metric(info: str) -> float:
    """Track task difficulty as a metric (0=easy, 1=medium, 2=hard)."""
    parsed = json.loads(info) if isinstance(info, str) else info
    difficulty_map = {"easy": 0.0, "medium": 1.0, "hard": 2.0}
    return difficulty_map.get(parsed.get("difficulty", ""), -1.0)


# ─── Environment ─────────────────────────────────────────────────────────

class OpenRCAEnv(vf.StatefulToolEnv):
    """OpenRCA environment for root cause analysis of software failures.

    Provides a persistent Python execution environment with tools for
    analyzing telemetry data (metrics, traces, logs) from microservice
    systems to identify root causes of failures.
    """

    _exec_lock = threading.Lock()

    def __init__(self, data_dir: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.data_dir = os.path.abspath(data_dir)
        self.add_tool(self.execute_python, args_to_skip=["namespace"])
        self.add_tool(self.list_directory)

    async def setup_state(self, state: vf.State, **kwargs: object) -> vf.State:
        """Initialize a persistent Python namespace for each rollout."""
        namespace: dict[str, object] = {}
        setup_code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "import os\n"
            "import json\n"
            "import re\n"
            "from datetime import datetime, timedelta\n"
            "import pytz\n"
            "\n"
            "pd.set_option('display.width', 427)\n"
            "pd.set_option('display.max_columns', 10)\n"
            "pd.set_option('display.max_rows', 50)\n"
            "\n"
            f"DATA_DIR = {self.data_dir!r}\n"
        )
        exec(setup_code, namespace)  # noqa: S102
        state["namespace"] = namespace
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, object],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: object,
    ) -> dict[str, object]:
        """Inject the persistent namespace into execute_python calls."""
        if tool_name == "execute_python":
            tool_args["namespace"] = state["namespace"]
        return tool_args

    async def execute_python(self, code: str, namespace: dict[str, object]) -> str:
        """Execute Python code in a persistent IPython-like environment.

        The environment has pandas, numpy, os, json, datetime, and pytz
        pre-imported. Variables persist across calls within the same rollout.
        The variable DATA_DIR contains the path to the telemetry data root.

        Args:
            code: Python code to execute. The last expression's value is
                  returned automatically (like in Jupyter/IPython).

        Returns:
            The execution output including printed text and the last
            expression's value, or an error message if execution fails.
        """

        def _run() -> str:
            with OpenRCAEnv._exec_lock:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                captured = StringIO()
                sys.stdout = captured
                sys.stderr = captured

                try:
                    tree = ast.parse(code)
                    last_value = None

                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last_expr = tree.body.pop()
                        if tree.body:
                            module = ast.Module(
                                body=tree.body, type_ignores=[]
                            )
                            exec(  # noqa: S102
                                compile(module, "<code>", "exec"), namespace
                            )
                        last_value = eval(  # noqa: S307
                            compile(
                                ast.Expression(body=last_expr.value),
                                "<code>",
                                "eval",
                            ),
                            namespace,
                        )
                    else:
                        exec(  # noqa: S102
                            compile(tree, "<code>", "exec"), namespace
                        )

                    output = captured.getvalue()
                    if last_value is not None:
                        value_str = str(last_value)
                        if output:
                            output = output.rstrip("\n") + "\n" + value_str
                        else:
                            output = value_str

                    if not output or not output.strip():
                        return "Code executed successfully (no output)."

                    if len(output) > MAX_OUTPUT_CHARS:
                        output = (
                            output[:MAX_OUTPUT_CHARS]
                            + "\n\n... [Output truncated. Use .head() or more "
                            "specific queries to limit output.]"
                        )

                    return output.strip()
                except Exception as e:
                    output = captured.getvalue()
                    error_msg = f"{type(e).__name__}: {e}"
                    if output and output.strip():
                        return f"{output.rstrip()}\n\n{error_msg}"
                    return error_msg
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run),
                timeout=EXEC_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return (
                f"Error: Code execution timed out after "
                f"{EXEC_TIMEOUT_SECONDS} seconds. Simplify your query or "
                f"process data in smaller chunks."
            )

    async def list_directory(self, path: str) -> str:
        """List files and directories within the telemetry data directory.

        Args:
            path: Relative path from the data root directory
                  (e.g., "Bank/telemetry/2021_03_05/metric").

        Returns:
            A listing of files and directories at the specified path.
        """

        def _list() -> str:
            full_path = os.path.normpath(os.path.join(self.data_dir, path))

            if not full_path.startswith(self.data_dir):
                return "Error: Access denied. Path must be within the data directory."

            if not os.path.exists(full_path):
                return f"Error: Path does not exist: {path}"

            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                return f"File: {path} ({size:,} bytes)"

            entries = sorted(os.listdir(full_path))
            if not entries:
                return "Empty directory."

            result = []
            for entry in entries:
                entry_path = os.path.join(full_path, entry)
                if os.path.isdir(entry_path):
                    result.append(f"[DIR]  {entry}/")
                else:
                    size = os.path.getsize(entry_path)
                    result.append(f"[FILE] {entry} ({size:,} bytes)")

            return "\n".join(result)

        return await asyncio.to_thread(_list)


# ─── Dataset builder ─────────────────────────────────────────────────────

def build_dataset(data_dir: str, systems: list[str]) -> Dataset:
    """Build a Verifiers Dataset from OpenRCA query CSV files.

    Each row becomes a rollout with a prompt containing the system-specific
    telemetry schema, root cause candidates, and the task instruction.

    Args:
        data_dir: Path to the OpenRCA dataset root directory.
        systems: List of system names to include.

    Returns:
        A HuggingFace Dataset with prompt, answer, and info columns.

    Raises:
        FileNotFoundError: If query CSV files are not found.
    """
    import pandas as pd

    rows = []
    for system in systems:
        query_path = os.path.join(data_dir, system, "query.csv")
        if not os.path.exists(query_path):
            raise FileNotFoundError(
                f"Query file not found at {query_path}. "
                f"Please download the OpenRCA dataset from "
                f"https://drive.google.com/drive/folders/"
                f"1wGiEnu4OkWrjPxfx5ZTROnU37-5UDoPM "
                f"and place it in '{data_dir}/'."
            )

        df = pd.read_csv(query_path)
        info = SYSTEM_INFO[system]

        for _, row in df.iterrows():
            instruction = row["instruction"]
            task_index = row["task_index"]
            scoring_points = row["scoring_points"]

            task_id = int(task_index.split("_")[1])
            if task_id <= 3:
                difficulty = "easy"
            elif task_id <= 6:
                difficulty = "medium"
            else:
                difficulty = "hard"

            user_content = (
                f"{info['schema']}\n\n"
                f"{info['candidates']}\n\n"
                f"## TASK\n\n"
                f"{instruction}\n\n"
                f"Use the `execute_python` tool to write and run Python code "
                f"for analyzing the telemetry data. Use the `list_directory` "
                f"tool to explore available data files. The `DATA_DIR` variable "
                f"in the Python environment points to the telemetry data root."
            )

            rows.append(
                {
                    "prompt": [{"role": "user", "content": user_content}],
                    "answer": scoring_points,
                    "info": json.dumps(
                        {
                            "system": system,
                            "task_index": task_index,
                            "difficulty": difficulty,
                        }
                    ),
                }
            )

    return Dataset.from_list(rows)


# ─── Entry point ─────────────────────────────────────────────────────────

def load_environment(
    data_dir: str = "dataset",
    systems: list[str] | None = None,
    max_turns: int = 30,
    num_examples: int = -1,
) -> vf.Environment:
    """Load the OpenRCA root cause analysis environment.

    Args:
        data_dir: Path to the OpenRCA dataset directory containing system
            subdirectories (Bank/, Market/, Telecom/) with query.csv and
            telemetry data.
        systems: List of systems to include. Options: "Bank",
            "Market/cloudbed-1", "Market/cloudbed-2", "Telecom".
            Defaults to all systems.
        max_turns: Maximum number of tool-use turns per rollout.
        num_examples: Number of examples to use. -1 for all.

    Returns:
        A Verifiers Environment ready for evaluation or training.
    """
    if systems is None:
        systems = list(ALL_SYSTEMS)

    for system in systems:
        if system not in SYSTEM_INFO:
            raise ValueError(
                f"Unknown system '{system}'. "
                f"Valid systems: {list(SYSTEM_INFO.keys())}"
            )

    dataset = build_dataset(data_dir, systems)

    if 0 < num_examples < len(dataset):
        dataset = dataset.select(range(num_examples))

    rubric = vf.Rubric(funcs=[openrca_score])
    rubric.add_metric(difficulty_metric)

    env = OpenRCAEnv(
        data_dir=data_dir,
        dataset=dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
    )

    return env
