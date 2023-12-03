import datetime
import psycopg2
from models.rfp import list_rfps
from models.agent import Agent

import os
import sys

module_path = "."
sys.path.append(os.path.abspath(module_path))


class Dao():
    conn = psycopg2.connect(database="procureai",
                            host="localhost",
                            user="postgres",
                            password="pYCKjpc474t26d5!!!",
                            port="15432")


def main():

    dao = Dao()
    rfps = list_rfps(dao.conn)

    # for r in rfps:
    #     print("Clearing rfp entries")
    #     r.clear_rfp(r.id)
    #     criteria_id = r.insert_criteria(r.id, "this is the criteria", 100)
    #     print(f"inserted criteria {criteria_id}")
    #     question_id = r.insert_question(
    #         criteria_id, f"another question {datetime.datetime.now()}")
    #     print(f"inserted question {question_id}")

    #     questions = r.list_questions()
    #     for q in questions:
    #         print(q)

    procurement_specialist_agent = Agent()
    # Agent - Generate criteria for sample rfp
    sample_rfp = rfps[0]
    procurement_specialist_agent.read_rfp(sample_rfp)
    procurement_specialist_agent.generate_criteria()

    # Agent - Generate questions for vendor


if __name__ == "__main__":
    main()
