import json
#s = json.__file__
#print json.__file__
import numpy as np 


global r_matrix
r_matrix = np.matrix([[0,0,0,0,0,0,0,0,0,0,0,10,20,30,40,50,60,70,80,90,100],
                      [0,0,0,0,0,0,0,0,0,10,20,30,40,50,60,70,80,90,00,90,80],
                      [0,0,0,0,0,0,0,10,20,30,40,50,60,70,80,90,100,90,80,70,60],
                      [0,0,0,0,0,10,20,30,40,50,60,70,80,90,100,90,80,70,60,50,40],
                      [0,0,0,10,20,30,40,50,60,70,80,90,100,90,80,70,60,50,40,30,20],
                      [0,10,20,30,40,50,60,70,80,90,100,90,80,70,60,50,40,30,20,10,0],
                      [20,30,40,50,60,70,80,90,100,90,80,70,60,50,40,30,20,10,0,0,0],
                      [40,50,60,70,80,90,100,90,80,70,60,50,40,30,20,10,0,0,0,0,0],
                      [60,70,80,90,100,90,80,70,60,50,40,30,20,10,0,0,0,0,0,0,0],
                      [80,90,100,90,80,70,60,50,40,30,20,10,0,0,0,0,0,0,0,0,0],
                      [100,90,80,70,60,50,40,30,20,10,0,0,0,0,0,0,0,0,0,0,0]])

print(r_matrix)

print(np.argmax(r_matrix[1,:]))
print(np.amax(r_matrix[1,:]))


book={}

book['hi'] = {
    'name':'hi',
    'address':'1 red street, NY',
    'phone':989898989
}

book['hello'] = {
    'name':'hello',
    'address':'1 red street, NY',
    'phone':989898989
}


q = np.zeros([11,21])

a = np.matrix([[1,2,3],[2,3,4]])

print(a)

print(q)

s=json.dumps(book)

with open("/home/pi/Desktop/q_matrix.txt","w") as f:
    f.write(s)

with open("/home/pi/Desktop/q_matrix.json","w") as json_write:
    json.dump(q.tolist(), json_write, indent=4)

with open("/home/pi/Desktop/q_matrix.json") as json_read:
    a = json.load(json_read)
print(np.array(a))


