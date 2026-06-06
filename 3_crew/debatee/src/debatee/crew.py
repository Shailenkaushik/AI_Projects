from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class Debatee():
    """Debatee crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def debater(self) -> Agent:
        return Agent(
            config=self.agents_config['debater'], # type: ignore[index]
            verbose=True
        )

    @agent
    def judge(self) -> Agent:
        return Agent(
            config=self.agents_config['judge'], # type: ignore[index]
            verbose=True
        )

    @task
    def favour_debater_task(self) -> Task:
        return Task(
            config=self.tasks_config['favour_debater_task'],
            output_file='favour_debater_report.md' # type: ignore[index]
        )

    @task
    def opposition_debater_task(self) -> Task:
        return Task(
            config=self.tasks_config['opposition_debater_task'], # type: ignore[index]
            output_file='opposition_debater_report.md'
        )

    @task
    def judge_task(self) -> Task:
        return Task(
            config=self.tasks_config['judge_task'], # type: ignore[index]
            output_file='judge_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Debatee crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
