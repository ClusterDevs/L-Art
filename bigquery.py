import uuid, os

from google.cloud import bigquery

def query_metart():
    client = bigquery.Client(project="change-to-your-project-id")

    query_job = client.query("""
        
        SELECT department, culture, link_resource
        FROM `bigquery-public-data.the_met.objects`
        WHERE culture IS NOT NULL""")

    results = query_job.result()  

    for row in results:
        print(row[0:3])

if __name__ == '__main__':
    query_metart()

