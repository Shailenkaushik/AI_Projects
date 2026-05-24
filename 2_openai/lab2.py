import os
import asyncio
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

async def invoke_agent(agent_instance: Agent, user_id: str, session_id: str, query: str) -> str:
    """
    Initializes an isolated runner for a specific agent, safeguards session state,
    and consumes the asynchronous event stream to pull the final response.
    """
    # 1. Initialize the memory-based runner for this specific agent
    runner = InMemoryRunner(agent=agent_instance)
    
    # 2. Pre-create the session container for this chat track
    await runner.session_service.create_session(
        app_name="parallel_agent_orchestrator",
        user_id=user_id,
        session_id=session_id
    )
    
    # 3. Construct the structured content object required by the ADK runtime
    user_content = types.Content(
        role="user",
        parts=[types.Part(text=query)]
    )
    
    final_text = "No response generated."
    
    try:
        # 4. Stream processing blocks from the runner
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content
        ):
            # Capture the compiled complete response instead of ongoing stream pieces
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break
    except Exception as e:
        final_text = f"An error occurred while running {agent_instance.name}: {str(e)}"
        
    return final_text


async def main():
    topic = "The impact of quantum computing on modern distributed database engines."
    print(f"Configuring specialized agents for topic: '{topic}'\n")
    
    # Create distinct specialized agents running on Gemini
    infrastructure_expert = Agent(
        name="InfrastructureExpert",
        model="gemini-2.5-flash",
        instruction="You are a core systems engineer. Focus on raw compute changes, indexing overheads, and hardware timing differences."
    )
    
    security_analyst = Agent(
        name="SecurityAnalyst",
        model="gemini-2.5-flash",
        instruction="You are a data security officer. Focus on post-quantum cryptography migrations, network handshake overhauls, and vector space protection."
    )
    
    cost_strategist = Agent(
        name="CostStrategist",
        model="gemini-2.5-flash",
        instruction="You are a financial cloud architect. Focus on infrastructure costs, rollout timelines, and early adopters value tracking."
    )

    user_id = "developer_user_123"
    
    # Map out separate jobs. We provide unique session IDs so their histories do not tangle.
    tasks = [
        invoke_agent(infrastructure_expert, user_id, "session_infra_99", f"Analyze this: {topic}"),
        invoke_agent(security_analyst, user_id, "session_sec_99", f"Analyze this: {topic}"),
        invoke_agent(cost_strategist, user_id, "session_cost_99", f"Analyze this: {topic}")
    ]
    
    print("Launching all agents concurrently via asyncio.gather()...")
    responses = await asyncio.gather(*tasks)
    print("All tasks completed successfully.\n")
    
    # Print individual results
    print("=" * 70)
    print(f"==> [1] INFRASTRUCTURE EXPERT EVALUATION:\n{responses[0]}\n")
    print("=" * 70)
    print(f"==> [2] SECURITY ANALYST EVALUATION:\n{responses[1]}\n")
    print("=" * 70)
    print(f"==> [3] COST STRATEGIST EVALUATION:\n{responses[2]}\n")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())