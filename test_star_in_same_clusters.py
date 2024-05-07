import json
import os

'''

A test file on the cluster_info file to see how many times the same star appears in different clusters

'''
foldername = "sigma5_rotated_border=0.2_only-sigma7"
jsonfile = "ALMA_DCT_dim_reduction=t-SNE_k-Means_n_clusters=70.json"
file_path= os.path.join("outputs","clusterinfo", foldername, jsonfile)
with open(file_path, "r") as json_file:
        cluster_info = json.load(json_file)

object_count = {}

for cluster_index, cluster in cluster_info.items():

    #Iterate through each item in the cluster
    for item in cluster:
        # Get the object name, shape before zooma nd rotationfactor
        object_name = item['info']['object']
        shape_before = item['info']['shape_before_resize']
        rotation_factor = item['info']['rotation_factor']
        if object_name in object_count:
            #only have one of these commented out at any time
            object_count[object_name].append({
                'cluster_index': cluster_index,
                'rotation_factor': rotation_factor,
                'shape_before': shape_before
            })
            #object_count[object_name].append([cluster_index])
        else:
             #only have one of these commented out at any time
             object_count[object_name] = [{
                'cluster_index': cluster_index,
                'rotation_factor': rotation_factor,
                'shape_before': shape_before
            }]#index with cluster number, shape before and rotation factor
             #object_count[object_name] = [[cluster_index]] #index with cluster number

#prints out how many different objects      
print(len(object_count))

#Prits out all objects with a list on all clusters it is in. Only use when we  just ahve a list of cluster indexes:
'''
for object_name in object_count:
    if len(object_count[object_name]) > 1:
         print(object_name , " i kluster: " , object_count[object_name])
'''
#Prints out all objects with  all clusters it is in and what has been done to the image
for object_name, clusters in object_count.items():
    print(object_name, " i kluster:")
    for cluster in clusters:
        cluster_index = cluster['cluster_index']
        rotation_factor = cluster['rotation_factor']
        shape_before = cluster['shape_before']
        print("- Kluster:", cluster_index, ", rotationsfaktor:", rotation_factor, ", storlek innan zoom:", shape_before)