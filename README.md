# Image_Clustering_Of_Protoplanetary_disks_ALMA
Kandidatarbete på Institutionen för Rymd, Geo- och Miljövetenskap på Chalmers Tekniska Högskola

Detektering, behandling och klustring av protoplanetära	skivor i ALMA-arkivet
Utveckling av en oövervakad maskininlärningsmodell

WILLIAM BERGQUIST, MARIE CARLANDER, THEA GRANSTRÖM, CONRAD OLSSON, VILHELM NILSSON THORSSON OCH HANNES ÖHMAN


# **Clustering.ipynb**


Datan som ska klustars finns i data\datasets lägg in namnet på den mappen som ska användas under variabeln "foldername".

Därefter kan kodeblocket "Modell Val" köras, välj attributextraheringsmetod, dimensionsreduceringsalgoritm och klustringsalgoritm och kör därefetr kodblocket igen.

Kodblocket under "Attributeextrahering" kan därefter köras. Samma sak med "Dimensionsreducering". Där kan paramentarna "verbose" och "perplexity" ändras efter tycke.

Därefter kan kodblocket under "Klustring" köras. Där kan "n_clusters" ändras om k-means valts som klsutings algoritm. "min_components" och "max_components" kan ändras om GMM valts som algoritm för att ändra på hur många kluster BIC testar. Slutligen om DBSCAN valts som klustingsalgoritm kan "epsilon" och "min_samples" justeras. Efter att detta block körts kommer en fil till outputs\collage\"foldername" där filnamnet består av vald attributsextarheringsmetod, dimensionsreduceringsmetod och val av klusteralgoritm samt val av parametrar på DBSCAN alternativt k-means. Liknande fil finns i \outputs\logs\"foldername" där alla filnamn för varje bild och kluster är utskrivet.

Därefter kan kodblocket under "Skapar jsonfil med information om klusterna" köras för att ladda ned informationen som finns i data\datasets\"foldername"\image_data.json om respektive objekt och sortera in dessa till vilket kluster de tillhör. Denna kan sedan användas för testning i test_star_in_same_clusters.py och heter samma som collage filen fast med .json och finns under \outputs\clusterinfo\"foldername"

Koden under Evaluation kan köras om k-means valts som klustringsalgoritm och man vill ha ett silhouette värde på hur bra klustringen blev.

Därefter finns ett kodblock som kan köras för att se plott på t-SNE och PCA.

# **test_star_in_same_clusters.py**
Först väljs vilken datamängd som ska användas och lägger in detta i "foldername". Därefter väljs vilke json-fil som ska testas, detta läggs in i jsonfile.

I denna testfil kan man välja om man vill

1. Bara ha information om i vilka kluster ett objekt som observerats flera gånger hamnar i för varje observation. 

Eller 

2. Om man vill veta för varje objekt, vilka kluster de är med i och vilken rotationsfaktor den genomgått och hur stor bilden var innan zoom för varje observation.

För alternativ 1 har man följade utkommenterat från koden:

object_count[object_name].append({

                'cluster_index': cluster_index,
                
                'rotation_factor': rotation_factor,
                
                'shape_before': shape_before
                
            })
            

object_count[object_name] = [{

                'cluster_index': cluster_index,
                
                'rotation_factor': rotation_factor,
                
                'shape_before': shape_before
                
            }]
            

for object_name, clusters in object_count.items():

    print(object_name, " i kluster:")
    
    for cluster in clusters:
    
        cluster_index = cluster['cluster_index']
        
        rotation_factor = cluster['rotation_factor']
        
        shape_before = cluster['shape_before']
        
        print("- Kluster:", cluster_index, ", rotationsfaktor:", rotation_factor, ", storlek innan zoom:", shape_before)

och låt resternade kod inte vara kommenterad. 

För alternativ 2 ska följande kod vara utkommenterad:

object_count[object_name].append([cluster_index])


object_count[object_name] = [[cluster_index]]



for object_name in object_count:

    if len(object_count[object_name]) > 1:
    
         print(object_name , " i kluster: " , object_count[object_name])

Resultatet av testerna skrivs ut i terminalen.
