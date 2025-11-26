from agent import TradeAgent
import time

def main():
    print("Starting Autonomous Trade Management Agent...")
    agent = TradeAgent()
    
    try:
        # Run one cycle immediately
        report = agent.run_cycle()
        agent.print_report(report)
        
        # In a real scenario, this would loop
        # while True:
        #     time.sleep(60)
        #     report = agent.run_cycle()
        #     agent.print_report(report)
            
    except KeyboardInterrupt:
        print("Agent stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
