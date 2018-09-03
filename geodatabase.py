import numpy as np
import psycopg2
from geoinput import GeoInput

class GeoDataBase(GeoInput):
    def __init__(self):
        conn = psycopg2.connect(host="localhost", database="DataLake", user="yuhang")

        cur = conn.cursor()
        cur.execute("select id, unique_id, ref_0001, ref_0001vs, ref_0010, ref_0100, ref_0100vs, thick, wii_clip_1,\
         ceno_clip_, x, y, z, class, age from yuhang.aem_shapefile where unique_id<160500")
        inputrecord = np.array(cur.fetchall())

        super(GeoDataBase,self).__init__(inputrecord)

