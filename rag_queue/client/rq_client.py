from redis import Redis
from rq import Queue

Queue = Queue(connection=Redis(
                host='localhost',
                port="6379"
 ))

