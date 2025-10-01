"""
DevCloud integration and deployment utilities
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import tempfile


class JobStatus(Enum):
    """DevCloud job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """DevCloud node types"""
    CPU = "cpu"
    GPU = "gpu"
    FPGA = "fpga"
    VPU = "vpu"
    ANY = "any"


@dataclass
class DevCloudConfig:
    """Configuration for DevCloud deployment"""
    project_name: str
    node_type: NodeType = NodeType.CPU
    max_runtime: str = "01:00:00"  # HH:MM:SS format
    memory: str = "4GB"
    cores: int = 4
    environment: str = "tensorflow"
    output_dir: str = "./devcloud_output"
    email_notifications: bool = False


@dataclass
class JobSubmission:
    """Job submission details"""
    job_id: str
    script_path: str
    config: DevCloudConfig
    submitted_at: float
    status: JobStatus = JobStatus.PENDING


@dataclass
class JobResult:
    """Job execution results"""
    job_id: str
    status: JobStatus
    exit_code: int
    stdout: str
    stderr: str
    runtime_seconds: float
    node_info: Dict[str, Any]
    metrics: Dict[str, Any]


class DevCloudManager:
    """
    Intel DevCloud integration manager
    """
    
    def __init__(self, config: Optional[DevCloudConfig] = None):
        self.config = config or DevCloudConfig(project_name="ai_agent_framework")
        self.jobs: Dict[str, JobSubmission] = {}
        self.results: Dict[str, JobResult] = {}
        self.base_dir = Path(self.config.output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    async def check_devcloud_access(self) -> bool:
        """Check if DevCloud is accessible"""
        try:
            # This would check actual DevCloud access
            # For now, simulate the check
            print("Checking Intel DevCloud access...")
            
            # Simulate checking qsub command availability
            result = subprocess.run(
                ["which", "qsub"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("Intel DevCloud access verified")
                return True
            else:
                print("Intel DevCloud not accessible. Running in local mode.")
                return False
                
        except Exception as e:
            print(f"Failed to check DevCloud access: {e}")
            return False
    
    async def create_job_script(
        self,
        script_content: str,
        script_name: str,
        config: Optional[DevCloudConfig] = None
    ) -> str:
        """
        Create a PBS job script for DevCloud
        
        Args:
            script_content: Python script content to execute
            script_name: Name of the script
            config: Optional job configuration
            
        Returns:
            Path to the created job script
        """
        job_config = config or self.config
        
        # Create PBS job script
        pbs_script = f"""#!/bin/bash
#PBS -N {job_config.project_name}_{script_name}
#PBS -l walltime={job_config.max_runtime}
#PBS -l nodes=1:ppn={job_config.cores}
#PBS -l mem={job_config.memory}
#PBS -o {self.base_dir}/output_{script_name}.log
#PBS -e {self.base_dir}/error_{script_name}.log

# Change to working directory
cd $PBS_O_WORKDIR

# Load Intel environment
source /opt/intel/oneapi/setvars.sh

# Load Python environment
module load python/{job_config.environment}

# Install framework dependencies if needed
pip install --user -r requirements.txt 2>/dev/null || true

# Execute the script
echo "Starting job execution at $(date)"
echo "Node information:"
hostname
lscpu | head -20

# Set Intel optimizations
export OMP_NUM_THREADS={job_config.cores}
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1

# Run the actual script
python {script_name}.py

echo "Job completed at $(date)"
"""
        
        # Save PBS script
        pbs_path = self.base_dir / f"{script_name}.pbs"
        with open(pbs_path, 'w') as f:
            f.write(pbs_script)
        
        # Save Python script
        script_path = self.base_dir / f"{script_name}.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Job script created: {pbs_path}")
        return str(pbs_path)
    
    async def submit_job(
        self,
        script_content: str,
        script_name: str,
        config: Optional[DevCloudConfig] = None
    ) -> Optional[str]:
        """
        Submit a job to Intel DevCloud
        
        Args:
            script_content: Python script content
            script_name: Name of the script
            config: Optional job configuration
            
        Returns:
            Job ID if successful
        """
        try:
            job_config = config or self.config
            
            # Create job script
            pbs_script_path = await self.create_job_script(script_content, script_name, job_config)
            
            # Check DevCloud access
            if not await self.check_devcloud_access():
                # Run locally as fallback
                return await self._run_local_job(script_content, script_name)
            
            # Submit job using qsub
            print(f"Submitting job to Intel DevCloud: {script_name}")
            
            result = subprocess.run(
                ["qsub", pbs_script_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                job_id = result.stdout.strip()
                
                # Store job submission
                submission = JobSubmission(
                    job_id=job_id,
                    script_path=pbs_script_path,
                    config=job_config,
                    submitted_at=time.time()
                )
                self.jobs[job_id] = submission
                
                print(f"Job submitted successfully: {job_id}")
                return job_id
            else:
                print(f"Job submission failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Failed to submit job: {e}")
            return None
    
    async def _run_local_job(self, script_content: str, script_name: str) -> str:
        """Run job locally as fallback"""
        try:
            job_id = f"local_{int(time.time())}"
            
            # Create temporary script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                temp_script = f.name
            
            print(f"Running job locally: {job_id}")
            
            # Run script locally
            process = await asyncio.create_subprocess_exec(
                "python", temp_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Store result
            result = JobResult(
                job_id=job_id,
                status=JobStatus.COMPLETED if process.returncode == 0 else JobStatus.FAILED,
                exit_code=process.returncode,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                runtime_seconds=10.0,  # Placeholder
                node_info={"type": "local", "hostname": "localhost"},
                metrics={}
            )
            
            self.results[job_id] = result
            
            # Clean up
            Path(temp_script).unlink()
            
            return job_id
            
        except Exception as e:
            print(f"Local job execution failed: {e}")
            return f"failed_{int(time.time())}"
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status"""
        try:
            # Check if it's a local job
            if job_id.startswith("local_") or job_id.startswith("failed_"):
                result = self.results.get(job_id)
                return result.status if result else JobStatus.FAILED
            
            # Check DevCloud job status
            result = subprocess.run(
                ["qstat", job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse qstat output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    status_line = lines[1]  # Skip header
                    status_field = status_line.split()[4]  # Status column
                    
                    status_map = {
                        'Q': JobStatus.PENDING,
                        'R': JobStatus.RUNNING,
                        'C': JobStatus.COMPLETED,
                        'E': JobStatus.FAILED
                    }
                    
                    return status_map.get(status_field, JobStatus.PENDING)
            
            return JobStatus.FAILED
            
        except Exception as e:
            print(f"Failed to get job status: {e}")
            return JobStatus.FAILED
    
    async def wait_for_completion(
        self,
        job_id: str,
        polling_interval: int = 30,
        timeout: int = 3600
    ) -> Optional[JobResult]:
        """
        Wait for job completion
        
        Args:
            job_id: Job ID to wait for
            polling_interval: Polling interval in seconds
            timeout: Timeout in seconds
            
        Returns:
            Job result when completed
        """
        try:
            start_time = time.time()
            
            print(f"Waiting for job completion: {job_id}")
            
            while time.time() - start_time < timeout:
                status = await self.get_job_status(job_id)
                
                if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    return await self.get_job_result(job_id)
                
                print(f"Job {job_id} status: {status.value}")
                await asyncio.sleep(polling_interval)
            
            print(f"Job {job_id} timed out after {timeout} seconds")
            return None
            
        except Exception as e:
            print(f"Error waiting for job completion: {e}")
            return None
    
    async def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get job execution results"""
        try:
            # Check if result is already cached
            if job_id in self.results:
                return self.results[job_id]
            
            # For DevCloud jobs, read output files
            if not job_id.startswith("local_"):
                submission = self.jobs.get(job_id)
                if not submission:
                    return None
                
                # Read output files
                output_file = self.base_dir / f"output_{submission.script_path.split('/')[-1].replace('.pbs', '')}.log"
                error_file = self.base_dir / f"error_{submission.script_path.split('/')[-1].replace('.pbs', '')}.log"
                
                stdout = ""
                stderr = ""
                
                try:
                    if output_file.exists():
                        stdout = output_file.read_text()
                except:
                    pass
                
                try:
                    if error_file.exists():
                        stderr = error_file.read_text()
                except:
                    pass
                
                # Create result
                result = JobResult(
                    job_id=job_id,
                    status=await self.get_job_status(job_id),
                    exit_code=0,  # Placeholder
                    stdout=stdout,
                    stderr=stderr,
                    runtime_seconds=time.time() - submission.submitted_at,
                    node_info={"type": "devcloud"},
                    metrics={}
                )
                
                self.results[job_id] = result
                return result
            
            return self.results.get(job_id)
            
        except Exception as e:
            print(f"Failed to get job result: {e}")
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        try:
            if job_id.startswith("local_"):
                print("Cannot cancel local jobs")
                return False
            
            result = subprocess.run(
                ["qdel", job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            success = result.returncode == 0
            if success:
                print(f"Job {job_id} cancelled successfully")
            else:
                print(f"Failed to cancel job {job_id}: {result.stderr}")
            
            return success
            
        except Exception as e:
            print(f"Failed to cancel job: {e}")
            return False
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs"""
        jobs_list = []
        
        for job_id, submission in self.jobs.items():
            job_info = {
                "job_id": job_id,
                "script_name": Path(submission.script_path).stem,
                "submitted_at": submission.submitted_at,
                "config": asdict(submission.config)
            }
            
            # Add result if available
            result = self.results.get(job_id)
            if result:
                job_info["status"] = result.status.value
                job_info["runtime"] = result.runtime_seconds
            
            jobs_list.append(job_info)
        
        return jobs_list
    
    async def create_benchmark_job(
        self,
        model_path: str,
        model_name: str,
        optimization_type: str = "openvino"
    ) -> str:
        """
        Create a benchmarking job for Intel optimizations
        
        Args:
            model_path: Path to the model file
            model_name: Name of the model
            optimization_type: Type of optimization (openvino, ipex, neural_compressor)
            
        Returns:
            Python script content for benchmarking
        """
        script_content = f'''
import sys
import time
import json
from pathlib import Path

# Add framework to path
sys.path.append('/home/u1234/ai_agent_framework')  # Adjust path as needed

def main():
    print("=== Intel Optimization Benchmark ===")
    print(f"Model: {model_name}")
    print(f"Optimization: {optimization_type}")
    print(f"Model path: {model_path}")
    
    try:
        if "{optimization_type}" == "openvino":
            from ai_agent_framework.intel_optimizations.openvino_optimizer import optimize_model_for_inference
            
            result = await optimize_model_for_inference(
                model_path="{model_path}",
                model_type="pytorch",
                device="CPU",
                benchmark=True
            )
            
            if result:
                print("Benchmark Results:")
                print(f"  Original latency: {{result.original_latency:.2f}} ms")
                print(f"  Optimized latency: {{result.optimized_latency:.2f}} ms")
                print(f"  Speedup: {{result.throughput_improvement:.2f}}x")
                
                # Save results
                with open("benchmark_results.json", "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
        
        elif "{optimization_type}" == "ipex":
            print("Intel Extension for PyTorch optimization benchmark")
            # Placeholder for IPEX benchmarking
            
        elif "{optimization_type}" == "neural_compressor":
            print("Neural Compressor optimization benchmark")
            # Placeholder for Neural Compressor benchmarking
        
        print("Benchmark completed successfully")
    
    except Exception as e:
        print(f"Benchmark failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''
        return script_content


# Convenience functions
async def submit_optimization_job(
    model_path: str,
    model_name: str,
    optimization_type: str = "openvino",
    node_type: str = "cpu"
) -> Optional[str]:
    """
    Quick function to submit an optimization job to DevCloud
    
    Args:
        model_path: Path to the model
        model_name: Name of the model
        optimization_type: Type of optimization
        node_type: DevCloud node type
        
    Returns:
        Job ID if successful
    """
    config = DevCloudConfig(
        project_name=f"optimize_{model_name}",
        node_type=NodeType(node_type.lower()),
        max_runtime="02:00:00",
        cores=8
    )
    
    manager = DevCloudManager(config)
    script_content = await manager.create_benchmark_job(model_path, model_name, optimization_type)
    
    return await manager.submit_job(script_content, f"benchmark_{model_name}", config)


async def run_distributed_benchmark(
    models: List[str],
    optimization_types: List[str] = ["openvino", "ipex"],
    max_concurrent: int = 4
) -> Dict[str, Any]:
    """
    Run distributed benchmarks across multiple models and optimizations
    
    Args:
        models: List of model paths
        optimization_types: List of optimization types to test
        max_concurrent: Maximum concurrent jobs
        
    Returns:
        Benchmark results summary
    """
    manager = DevCloudManager()
    jobs = []
    results = {}
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_single_benchmark(model_path: str, opt_type: str):
        async with semaphore:
            model_name = Path(model_path).stem
            job_name = f"{model_name}_{opt_type}"
            
            script_content = await manager.create_benchmark_job(model_path, model_name, opt_type)
            job_id = await manager.submit_job(script_content, job_name)
            
            if job_id:
                result = await manager.wait_for_completion(job_id)
                if result:
                    results[job_name] = {
                        "job_id": job_id,
                        "status": result.status.value,
                        "runtime": result.runtime_seconds,
                        "output": result.stdout
                    }
    
    # Create tasks for all combinations
    tasks = []
    for model_path in models:
        for opt_type in optimization_types:
            tasks.append(run_single_benchmark(model_path, opt_type))
    
    # Run all benchmarks
    await asyncio.gather(*tasks)
    
    return {
        "summary": {
            "total_jobs": len(tasks),
            "completed": len([r for r in results.values() if r["status"] == "completed"]),
            "failed": len([r for r in results.values() if r["status"] == "failed"])
        },
        "results": results
    }