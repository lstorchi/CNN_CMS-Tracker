
import tensorflow as tf

def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s %s' % (i, node.name, node.op, node.input))


# read frozen graph and display nodes                                                    
graph = tf.GraphDef()
with tf.gfile.Open('pixel_only_final.pb', 'r') as f:
    data = f.read()
    graph.ParseFromString(data)

display_nodes(graph.node)
