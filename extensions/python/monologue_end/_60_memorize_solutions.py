from helpers import plugins, errors
from helpers.extension import Extension
from usr.plugins.memory_cognee.helpers.memory import Memory, insert_with_simple_dedup
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

        db = await Memory.get(self.agent)

        log_item = self.agent.context.log.log(
            type="util",
            heading="Memorizing succesful solutions...",
        )

        task = DeferredTask(thread_name=THREAD_BACKGROUND)
        task.start_task(self.memorize, loop_data, log_item, db, cfg)
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

            log_item.update(content=solutions_json)

            if not solutions_json or not isinstance(solutions_json, str):
                log_item.update(heading="No response from utility model.")
                return

            solutions_json = solutions_json.strip()

            if not solutions_json:
                log_item.update(heading="Empty response from utility model.")
                return

            try:
                solutions = DirtyJson.parse_string(solutions_json)
            except Exception as e:
                log_item.update(heading=f"Failed to parse solutions response: {str(e)}")
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

            if use_consolidation:
                from usr.plugins.memory_cognee.helpers.memory_consolidation import create_memory_consolidator
                consolidator = create_memory_consolidator(
                    self.agent,
                    similarity_threshold=cfg.get("memory_recall_similarity_threshold", 0.7),
                    replace_similarity_threshold=replace_threshold,
                )

            for solution in solutions:
                if isinstance(solution, dict):
                    problem = solution.get("problem", "Unknown problem")
                    solution_text = solution.get("solution", "Unknown solution")
                    txt = f"# Problem\n {problem}\n# Solution\n {solution_text}"
                else:
                    txt = f"# Solution\n {str(solution)}"

                if use_consolidation:
                    await consolidator.process_new_memory(
                        new_memory=txt,
                        area=area,
                        metadata={"area": area},
                        log_item=log_item,
                    )
                else:
                    await insert_with_simple_dedup(db, txt, area, replace_threshold)

            log_item.update(
                result=f"{len(solutions)} solutions memorized.",
                heading=f"{len(solutions)} solutions memorized.",
            )

        except Exception as e:
            err = errors.format_error(e)
            self.agent.context.log.log(
                type="warning", heading="Memorize solutions extension error", content=err
            )
