#!/usr/bin/env python3

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
import threading


class COVIDLauncher:
    def __init__(self):
        self.api_process = None
        self.dashboard_process = None
        self.running = True

    def check_dependencies(self):
        """Check if required packages are installed"""
        required_packages = [
            'fastapi', 'uvicorn', 'dash', 'plotly', 'pandas',
            'requests', 'snowflake-connector-python', 'pymongo',
            'dash-bootstrap-components', 'scipy', 'scikit-learn'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print(f"Install with: pip install {' '.join(missing_packages)}")
            return False

        print("✅ All required packages are installed")
        return True

    def check_env_file(self):
        """Check if .env file exists and has required variables"""
        env_file = Path('.env')
        if not env_file.exists():
            print("⚠️  .env file not found. Creating template...")
            self.create_env_template()
            return False

        required_vars = [
            'SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD',
            'MONGODB_URI', 'API_BASE_URL'
        ]

        env_content = env_file.read_text()
        missing_vars = []

        for var in required_vars:
            if f'{var}=' not in env_content:
                missing_vars.append(var)

        if missing_vars:
            print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
            print("Please update your .env file with the required variables")
            return False

        print("✅ Environment file looks good")
        return True

    def create_env_template(self):
        """Create a template .env file"""
        template = """# COVID Dashboard Environment Variables

# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your_account.region
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=COVID19_EPIDEMIOLOGICAL_DATA
SNOWFLAKE_SCHEMA=PUBLIC

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=COVID_COMMENTS

# API Configuration
API_BASE_URL=http://localhost:8003

# Optional: Logging level
LOG_LEVEL=INFO
"""
        with open('.env', 'w') as f:
            f.write(template)
        print("📝 Created .env template file. Please fill in your credentials.")

    def start_api(self):
        """Start the FastAPI server"""
        print("🚀 Starting COVID API server on port 8003...")
        try:
            self.api_process = subprocess.Popen([
                sys.executable, 'api.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Wait a moment to check if it started successfully
            time.sleep(2)
            if self.api_process.poll() is None:
                print("✅ API server started successfully")
                return True
            else:
                stdout, stderr = self.api_process.communicate()
                print(f"❌ API server failed to start:")
                print(f"STDOUT: {stdout.decode()}")
                print(f"STDERR: {stderr.decode()}")
                return False
        except Exception as e:
            print(f"❌ Failed to start API server: {e}")
            return False

    def start_dashboard(self):
        """Start the Dash dashboard"""
        print("🎯 Starting COVID Dashboard on port 8050...")
        try:
            self.dashboard_process = subprocess.Popen([
                sys.executable, 'dashboard.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Wait a moment to check if it started successfully
            time.sleep(3)
            if self.dashboard_process.poll() is None:
                print("✅ Dashboard started successfully")
                return True
            else:
                stdout, stderr = self.dashboard_process.communicate()
                print(f"❌ Dashboard failed to start:")
                print(f"STDOUT: {stdout.decode()}")
                print(f"STDERR: {stderr.decode()}")
                return False
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            return False

    def monitor_processes(self):
        """Monitor running processes and restart if needed"""
        while self.running:
            time.sleep(5)

            # Check API process
            if self.api_process and self.api_process.poll() is not None:
                print("⚠️  API process stopped. Attempting restart...")
                if not self.start_api():
                    print("❌ Failed to restart API process")
                    break

            # Check Dashboard process
            if self.dashboard_process and self.dashboard_process.poll() is not None:
                print("⚠️  Dashboard process stopped. Attempting restart...")
                if not self.start_dashboard():
                    print("❌ Failed to restart Dashboard process")
                    break

    def stop_processes(self):
        """Stop all running processes"""
        print("\n🛑 Stopping all processes...")
        self.running = False

        if self.api_process:
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=5)
                print("✅ API server stopped")
            except subprocess.TimeoutExpired:
                self.api_process.kill()
                print("💀 API server force killed")
            except Exception as e:
                print(f"⚠️  Error stopping API: {e}")

        if self.dashboard_process:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=5)
                print("✅ Dashboard stopped")
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
                print("💀 Dashboard force killed")
            except Exception as e:
                print(f"⚠️  Error stopping Dashboard: {e}")

    def run(self):
        """Main run method"""
        print("🦠 COVID-19 Analytics Platform Launcher")
        print("=" * 50)

        # Pre-flight checks
        if not self.check_dependencies():
            return False

        if not self.check_env_file():
            return False

        # Start both services
        print("\n🚀 Starting both API and Dashboard services...")

        # Start API first
        if not self.start_api():
            return False

        # Wait for API to be ready, then start dashboard
        print("⏳ Waiting for API to be ready...")
        time.sleep(3)

        if not self.start_dashboard():
            self.stop_processes()
            return False

        # Print access information
        print("\n" + "=" * 50)
        print("🎉 COVID Analytics Platform is running!")
        print("=" * 50)
        print("📊 API Documentation: http://localhost:8003/docs")
        print("💾 API Health Check: http://localhost:8003/health")
        print("🎯 Dashboard: http://localhost:8050")
        print("\nPress Ctrl+C to stop all services")
        print("=" * 50)

        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()

        # Wait for interrupt
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_processes()

        return True


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal")
    launcher.running = False


if __name__ == "__main__":
    # Create launcher instance
    launcher = COVIDLauncher()

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Run the launcher
        if launcher.run():
            print("\n👋 COVID Analytics Platform stopped cleanly")
            sys.exit(0)
        else:
            print("\n💥 COVID Analytics Platform failed to start")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        launcher.stop_processes()
        sys.exit(1)