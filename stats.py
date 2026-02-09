import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from typing import Optional, Dict, Any, Tuple


RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
CPU_RE = re.compile(r"CPU\s+\[([^\]]+)\]")
GPU_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
TEMP_RE = re.compile(r"([A-Za-z0-9_]+)@([0-9.]+)C")
POWER_MW_RE = re.compile(r"([A-Za-z0-9_]+)\s+(\d+)mW")
POM_RE = re.compile(r"(POM_5V_[A-Za-z0-9_]+)\s+(\d+)/(\d+)")


def parse_tegrastats_line(line: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}

    ram_match = RAM_RE.search(line)
    if ram_match:
        data["ram_used_mb"] = float(ram_match.group(1))
        data["ram_total_mb"] = float(ram_match.group(2))

    cpu_match = CPU_RE.search(line)
    if cpu_match:
        vals = [int(v) for v in re.findall(r"(\d+)%", cpu_match.group(1))]
        if vals:
            data["cpu_util_pct"] = sum(vals) / len(vals)

    gpu_match = GPU_RE.search(line)
    if gpu_match:
        data["gpu_util_pct"] = float(gpu_match.group(1))

    temps: Dict[str, float] = {}
    for name, temp in TEMP_RE.findall(line):
        try:
            temps[name] = float(temp)
        except ValueError:
            continue
    if temps:
        data["temps_c"] = temps

    power: Dict[str, float] = {}
    for name, val in POWER_MW_RE.findall(line):
        power[name] = float(val)
    for name, inst, avg in POM_RE.findall(line):
        power[name] = float(inst)
        power["{0}_AVG".format(name)] = float(avg)
    if power:
        data["power_mw"] = power

    return data


class TegraStatsSampler:
    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = interval_ms
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.cpu_sum = 0.0
        self.cpu_count = 0
        self.gpu_sum = 0.0
        self.gpu_count = 0
        self.ram_used_sum = 0.0
        self.ram_count = 0
        self.ram_total_mb: Optional[float] = None
        self.temp_sums: Dict[str, float] = defaultdict(float)
        self.temp_counts: Dict[str, int] = defaultdict(int)
        self.power_sums: Dict[str, float] = defaultdict(float)
        self.power_counts: Dict[str, int] = defaultdict(int)
        self.sample_count = 0

    def start(self) -> bool:
        if not shutil.which("tegrastats"):
            return False
        try:
            self.process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError:
            return False

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return True

    def _worker(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            if self.stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            parsed = parse_tegrastats_line(line)
            if not parsed:
                continue
            with self.lock:
                self.sample_count += 1
                if "cpu_util_pct" in parsed:
                    self.cpu_sum += float(parsed["cpu_util_pct"])
                    self.cpu_count += 1
                if "gpu_util_pct" in parsed:
                    self.gpu_sum += float(parsed["gpu_util_pct"])
                    self.gpu_count += 1
                if "ram_used_mb" in parsed:
                    self.ram_used_sum += float(parsed["ram_used_mb"])
                    self.ram_count += 1
                if "ram_total_mb" in parsed:
                    self.ram_total_mb = float(parsed["ram_total_mb"])
                temps = parsed.get("temps_c")
                if isinstance(temps, dict):
                    for key, val in temps.items():
                        self.temp_sums[key] += float(val)
                        self.temp_counts[key] += 1
                power = parsed.get("power_mw")
                if isinstance(power, dict):
                    for key, val in power.items():
                        self.power_sums[key] += float(val)
                        self.power_counts[key] += 1

    def stop(self) -> None:
        self.stop_event.set()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread:
            self.thread.join(timeout=2)

    def averages(self) -> Dict[str, Any]:
        with self.lock:
            temps_avg = {
                k: self.temp_sums[k] / self.temp_counts[k]
                for k in self.temp_sums
                if self.temp_counts[k]
            }
            power_avg = {
                k: self.power_sums[k] / self.power_counts[k]
                for k in self.power_sums
                if self.power_counts[k]
            }
            return {
                "cpu_avg_pct": self.cpu_sum / self.cpu_count if self.cpu_count else None,
                "gpu_avg_pct": self.gpu_sum / self.gpu_count if self.gpu_count else None,
                "ram_avg_used_mb": self.ram_used_sum / self.ram_count if self.ram_count else None,
                "ram_total_mb": self.ram_total_mb,
                "temps_avg_c": temps_avg or None,
                "power_avg_mw": power_avg or None,
                "samples": self.sample_count,
            }


class PsutilSampler:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.cpu_sum = 0.0
        self.cpu_count = 0
        self.ram_used_sum = 0.0
        self.ram_count = 0
        self.ram_total_mb: Optional[float] = None
        self.sample_count = 0
        self.psutil = None

    def start(self) -> bool:
        try:
            import psutil
        except ImportError:
            return False
        self.psutil = psutil
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return True

    def _worker(self) -> None:
        assert self.psutil is not None
        self.psutil.cpu_percent(interval=None)
        while not self.stop_event.is_set():
            cpu = self.psutil.cpu_percent(interval=None)
            mem = self.psutil.virtual_memory()
            with self.lock:
                self.sample_count += 1
                self.cpu_sum += float(cpu)
                self.cpu_count += 1
                self.ram_used_sum += float(mem.used) / (1024 * 1024)
                self.ram_count += 1
                self.ram_total_mb = float(mem.total) / (1024 * 1024)
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def averages(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "cpu_avg_pct": self.cpu_sum / self.cpu_count if self.cpu_count else None,
                "gpu_avg_pct": None,
                "ram_avg_used_mb": self.ram_used_sum / self.ram_count if self.ram_count else None,
                "ram_total_mb": self.ram_total_mb,
                "temps_avg_c": None,
                "power_avg_mw": None,
                "samples": self.sample_count,
            }


def select_sampler(interval_ms: int) -> Tuple[Optional[object], str]:
    tegra = TegraStatsSampler(interval_ms)
    if tegra.start():
        return tegra, "tegrastats"
    psutil_sampler = PsutilSampler(interval_ms / 1000.0)
    if psutil_sampler.start():
        return psutil_sampler, "psutil"
    return None, "none"
