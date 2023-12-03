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

    procurement_specialist_agent = Agent()
    # Agent - Generate criteria for sample rfp
    sample_rfp = rfps[0]

    print("Reading RFPs...")
    procurement_specialist_agent.read_rfp(sample_rfp)

    print("Generate criteria and guide questions for vendors...")
    criteria = procurement_specialist_agent.generate_criteria()
    print("Criteria generated. Please review below before openning RFP to public.")
    for c in criteria:
        print(f"- {c['description']} {c['percentage']}%")
        print(c['questions'])

    # Agent - Generate questions for vendor
    print("Evaluate vendor submissions in responses folder...")
    procurement_specialist_agent.evaluate_responses(sample_rfp, criteria)
    print("Evaluation finished.")


if __name__ == "__main__":
    main()
