import sys
import discord
from discord.ext import commands
import subprocess
import asyncio
import os
import psutil
import signal
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Bot configuration
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
AUTHORIZED_USERS =  [int(os.getenv('AUTHORIZED_USERS'))] 

# Create bot instance
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Store running processes
running_processes = {}

def is_authorized(user_id):
    """Check if user is authorized to use bot commands"""
    return user_id in AUTHORIZED_USERS

@bot.event
async def on_ready():
    print(f'🤖 {bot.user} is now online!')
    print(f'Bot ID: {bot.user.id}')

# Store current working directory and venv per user
user_directories = {}
user_venvs = {}

@bot.command(name='activate_venv')
async def activate_venv(ctx, venv_path=None):
    """Activate a virtual environment"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    if venv_path is None:
        venv_path = "/home/anou/Documents/reinforcement_learning_thesis/.venv"
    
    # Check if venv exists
    activate_script = os.path.join(venv_path, "bin", "activate")
    if os.path.exists(activate_script):
        user_venvs[ctx.author.id] = venv_path
        await ctx.send(f"✅ Virtual environment activated: `{venv_path}`")
    else:
        await ctx.send(f"❌ Virtual environment not found: `{venv_path}`")

@bot.command(name='deactivate_venv')
async def deactivate_venv(ctx):
    """Deactivate current virtual environment"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    if ctx.author.id in user_venvs:
        del user_venvs[ctx.author.id]
        await ctx.send("✅ Virtual environment deactivated")
    else:
        await ctx.send("❌ No virtual environment active")

# Modify the existing cmd function
@bot.command(name='cmd')
async def execute_command(ctx, *, command):
    """Execute a terminal command"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    try:
        # Security check...
        dangerous_commands = ['rm -rf /', 'sudo rm -rf', 'dd if=', 'mkfs', 'fdisk']
        if any(dangerous in command.lower() for dangerous in dangerous_commands):
            await ctx.send("❌ Dangerous command blocked for safety.")
            return
        
        # Get user's working directory
        working_dir = user_directories.get(ctx.author.id, os.getcwd())
        
        # Prepare command with venv activation if needed
        venv_path = user_venvs.get(ctx.author.id)
        if venv_path:
            activate_script = os.path.join(venv_path, "bin", "activate")
            command = f"source {activate_script} && {command}"
        
        # Execute command
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30,
            cwd=working_dir
        )
        
        # Rest of the function remains the same...
        output = result.stdout if result.stdout else "Command executed successfully (no output)"
        error = result.stderr if result.stderr else ""
        
        if len(output) > 1900:
            output = output[:1900] + "... (truncated)"
        if len(error) > 1900:
            error = error[:1900] + "... (truncated)"
        
        response = f"```bash\n$ {command}\n{output}"
        if error:
            response += f"\nErrors:\n{error}"
        response += "```"
        
        await ctx.send(response)
        
    except subprocess.TimeoutExpired:
        await ctx.send("❌ Command timed out (30s limit)")
    except Exception as e:
        await ctx.send(f"❌ Error executing command: {e}")

@bot.command(name='start_optuna')
async def start_optuna(ctx, study_name="go2-obstacles", n_trials=50):
    """Start Optuna optimization"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    try:
        # Check if already running
        if 'optuna' in running_processes:
            await ctx.send("❌ Optuna is already running!")
            return
        
        command = f"cd /home/anou/Documents/reinforcement_learning_thesis/go2_locomotion && python test_optuna.py -e {study_name} -T {n_trials}"
        
        # Start process in background
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        running_processes['optuna'] = process
        await ctx.send(f"🚀 Started Optuna optimization: {study_name} ({n_trials} trials)\nPID: {process.pid}")
        
    except Exception as e:
        await ctx.send(f"❌ Failed to start Optuna: {e}")

@bot.command(name='stop_optuna')
async def stop_optuna(ctx):
    """Stop running Optuna process"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    try:
        if 'optuna' not in running_processes:
            await ctx.send("❌ No Optuna process running.")
            return
        
        process = running_processes['optuna']
        process.terminate()
        del running_processes['optuna']
        
        await ctx.send("⏹️ Optuna process stopped.")
        
    except Exception as e:
        await ctx.send(f"❌ Error stopping Optuna: {e}")

@bot.command(name='status')
async def system_status(ctx):
    """Get system status"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    try:
        # Check if optuna is running
        optuna_running = 'optuna' in running_processes and running_processes['optuna'].poll() is None
        
        # System info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check for specific processes
        processes = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        optuna_in_ps = 'test_optuna.py' in processes.stdout
        
        status_msg = f"""```
🖥️  System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔄 Processes:
   Optuna (bot):     {'✅ Running' if optuna_running else '❌ Stopped'}
   Optuna (system):  {'✅ Running' if optuna_in_ps else '❌ Stopped'}

📊 Resources:
   CPU Usage:    {cpu_percent}%
   Memory:       {memory.percent}% ({memory.used//1024//1024}MB / {memory.total//1024//1024}MB)
   Disk Usage:   {disk.percent}% ({disk.used//1024//1024//1024}GB / {disk.total//1024//1024//1024}GB)
```"""
        
        await ctx.send(status_msg)
        
    except Exception as e:
        await ctx.send(f"❌ Error getting status: {e}")

@bot.command(name='logs')
async def get_logs(ctx, lines=20):
    """Get recent log entries"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    try:
        # Get recent logs from your training
        log_path = "/home/anou/Documents/reinforcement_learning_thesis/go2_locomotion/logs"
        
        if os.path.exists(log_path):
            # Find most recent log directory
            log_dirs = [d for d in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, d))]
            if log_dirs:
                latest_log = max(log_dirs, key=lambda x: os.path.getctime(os.path.join(log_path, x)))
                log_file = os.path.join(log_path, latest_log, "summaries.txt")
                
                if os.path.exists(log_file):
                    result = subprocess.run(['tail', f'-{lines}', log_file], capture_output=True, text=True)
                    logs = result.stdout[:1900] if result.stdout else "No recent logs found"
                    await ctx.send(f"```\nRecent logs from {latest_log}:\n{logs}\n```")
                else:
                    await ctx.send("❌ Log file not found")
            else:
                await ctx.send("❌ No log directories found")
        else:
            await ctx.send("❌ Log directory not found")
            
    except Exception as e:
        await ctx.send(f"❌ Error getting logs: {e}")

@bot.command(name='kill')
async def kill_process(ctx, process_name):
    """Kill a process by name"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    try:
        result = subprocess.run(['pkill', '-f', process_name], capture_output=True, text=True)
        if result.returncode == 0:
            await ctx.send(f"✅ Killed processes matching: {process_name}")
        else:
            await ctx.send(f"❌ No processes found matching: {process_name}")
            
    except Exception as e:
        await ctx.send(f"❌ Error killing process: {e}")

@bot.command(name='help_terminal')
async def help_terminal(ctx):
    """Show available commands"""
    help_text = """```
🤖 Discord Terminal Bot Commands:

!cmd <command>              - Execute any terminal command
!start_optuna [name] [trials] - Start Optuna optimization  
!stop_optuna               - Stop running Optuna process
!status                    - Show system status & resources
!logs [lines]              - Show recent training logs (default: 20)
!kill <process_name>       - Kill process by name
!help_terminal             - Show this help message

Examples:
!cmd ls -la
!cmd nvidia-smi
!start_optuna my_study 100
!logs 50
!kill test_optuna.py
```"""
    await ctx.send(help_text)

@bot.command(name='check_movement')
async def check_movement(ctx):
    """Check if robot is moving in current training"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    try:
        # Check recent logs for velocity
        command = "grep -i 'vel\|reward' /home/anou/Documents/reinforcement_learning_thesis/go2_locomotion/logs/*/summaries.txt | tail -10"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            await ctx.send(f"```\nRecent velocity/reward data:\n{result.stdout[:1900]}\n```")
        else:
            await ctx.send("❌ No movement data found in logs")
            
    except Exception as e:
        await ctx.send(f"❌ Error checking movement: {e}")

@bot.command(name='restart_bot')
async def restart_bot(ctx):
    """Restart the Discord bot"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    await ctx.send("🔄 Restarting bot...")
    await bot.close()
    os.execv(sys.executable, ['python'] + sys.argv)

@bot.command(name='start_training')
async def start_training(ctx, params_pkl=None, study_name=None):
    """Start training process"""
    if not is_authorized(ctx.author.id):
        await ctx.send("❌ You are not authorized to use this bot.")
        return
    
    try:
        # Check if already running
        if 'training' in running_processes:
            await ctx.send("❌ Training is already running!")
            return

        command = f"cd /home/anou/Documents/reinforcement_learning_thesis/go2_locomotion && python train_from_script.py --study_name {study_name} --params_pkl {params_pkl}"

        # Start process in background
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        running_processes['training'] = process
        await ctx.send(f"🚀 Started training: python start_from_script.py --study_name {study_name} --params_pkl {params_pkl}\nPID: {process.pid}")
        
    except Exception as e:
        await ctx.send(f"❌ Failed to start training: {e}")

# Run the bot
if __name__ == "__main__":
    bot.run(BOT_TOKEN)