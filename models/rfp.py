import os
from os import listdir
from os.path import isfile, join


class Rfp:
    path = ""
    id = ""

    def __init__(self, id, conn):
        self.conn = conn
        cwd = os.getcwd()

        self.path = f"{cwd}/rfp/{id}"
        self.id = id

    def list_criteria(self):
        cursor = self.conn.cursor()
        query = "select * from criteria"

        cursor.execute(query)
        records = cursor.fetchall()

        return records

    def list_questions(self):
        cursor = self.conn.cursor()
        query = """
            SELECT criteria.id, criteria.rfp_id, criteria.description, criteria.percent, questions.id, questions.question
            FROM public.criteria
            RIGHT JOIN public.questions
            ON criteria.id = questions.criteria_id
            WHERE criteria.rfp_id=%s;
        """
        cursor.execute(query, (self.id,))
        records = cursor.fetchall()

        return records

    def insert_question(self, criteria_id,  question):
        cursor = self.conn.cursor()
        query = """
            INSERT INTO public.questions(
                criteria_id, question)
                VALUES (%s, %s) RETURNING id;
        """

        cursor.execute(query, (criteria_id, question,))
        id = cursor.fetchone()[0]
        self.conn.commit()
        return id

    def insert_criteria(self, rfp_id,  description, questions, percent):
        cursor = self.conn.cursor()
        query = """
            INSERT INTO public.criteria(
                rfp_id, description, questions, percent)
                VALUES (%s, %s, %s, %s) RETURNING id;
        """

        cursor.execute(query, (rfp_id, description, questions, percent))
        id = cursor.fetchone()[0]
        self.conn.commit()
        return id

    def clear_rfp(self, rfp_id):
        cursor = self.conn.cursor()
        query = """
            DELETE FROM public.criteria WHERE rfp_id=%s;
        """
        cursor.execute(query, (rfp_id,))
        self.conn.commit()
        return id


def list_rfps(conn):
    cwd = os.getcwd()
    rdp_dir = f"{cwd}/rfp"
    onlyfiles = [Rfp(f, conn)
                 for f in listdir(rdp_dir) if isfile(join(rdp_dir, f))]
    return onlyfiles
