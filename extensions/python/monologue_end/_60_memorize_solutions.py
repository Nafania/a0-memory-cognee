from helpers import plugins, errors
from helpers.extension import Extension
from usr.plugins.memory_cognee.helpers.memory import Memory
from usr.plugins.memory_cognee.helpers.llm_json import (
    format_llm_json_for_log,
    parse_llm_json_response,
)
from usr.plugins.memory_cognee.helpers.memory_write_worker import MemoryWriteWorker
from helpers.dirty_json import DirtyJson
from agent import LoopData
from helpers.log import LogItem
from helpers.defer import DeferredTask, THREAD_BACKGROUND


class MemorizeSolutions(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        cfg = plugins.get_plugin_config("memory_cognee", self.agent)
        if not cfg:
            return

        if not cfg["memory_memorize_enabled"]:
            return

        log_item = self.agent.context.log.log(
            type="util",
            heading="Memorizing succesful solutions...",
        )

        task = DeferredTask(thread_name=THREAD_BACKGROUND)
        task.start_task(self.memorize, loop_data, log_item, None, cfg)
        return task

    async def memorize(self, loop_data: LoopData, log_item: LogItem, db: Memory, cfg: dict, **kwargs):
        try:
            system = self.agent.read_prompt("memory.solutions_sum.sys.md")
            msgs_text = self.agent.concat_messages(self.agent.history)

            solutions_json = await self.agent.call_utility_model(
                system=system,
                message=msgs_text,
                background=True,
            )

            if not solutions_json or not isinstance(solutions_json, str):
                log_item.update(heading="No response from utility model.")
                return

            solutions_json = solutions_json.strip()

            if not solutions_json:
                log_item.update(heading="Empty response from utility model.")
                return

            try:
                solutions = parse_llm_json_response(solutions_json, DirtyJson.parse_string)
            except Exception as e:
                log_item.update(
                    heading=f"Failed to parse solutions response: {str(e)}",
                    content=solutions_json,
                )
                return

            if solutions is None:
                log_item.update(heading="No valid solutions found in response.")
                return

            if not isinstance(solutions, list):
                if isinstance(solutions, (str, dict)):
                    solutions = [solutions]
                else:
                    log_item.update(heading="Invalid solutions format received.")
                    return

            log_item.update(content=format_llm_json_for_log(solutions))

            if not isinstance(solutions, list) or len(solutions) == 0:
                log_item.update(heading="No successful solutions to memorize.")
                return
            else:
                solutions_txt = "\n\n".join([str(solution) for solution in solutions]).strip()
                log_item.update(
                    heading=f"{len(solutions)} successful solutions to memorize.", solutions=solutions_txt
                )

            use_consolidation = cfg.get("memory_memorize_consolidation", False)
            replace_threshold = cfg.get("memory_memorize_replace_threshold", 0.9)
            area = Memory.Area.SOLUTIONS.value
            worker = MemoryWriteWorker.get_instance()
            queued = 0
            for solution in solutions:
                if isinstance(solution, dict):
                    problem = solution.get("problem", "Unknown problem")
                    solution_text = solution.get("solution", "Unknown solution")
                    txt = f"# Problem\n {problem}\n# Solution\n {solution_text}"
                else:
                    txt = f"# Solution\n {str(solution)}"

                queued = worker.enqueue(
                    agent=self.agent,
                    text=txt,
                    area=area,
                    metadata={"area": area},
                    cfg=cfg,
                    use_consolidation=use_consolidation,
                    replace_threshold=replace_threshold,
                    similarity_threshold=cfg.get("memory_recall_similarity_threshold", 0.7),
                )

            log_item.update(
                result=f"{len(solutions)} solutions queued for memory write.",
                heading=f"{len(solutions)} solutions queued for memory write.",
                memory_ids=[],
                queued_memory_count=len(solutions),
                memory_write_queue_size=queued,
            )

        except Exception as e:
            err = errors.format_error(e)
            self.agent.context.log.log(
                type="warning", heading="Memorize solutions extension error", content=err
            )
