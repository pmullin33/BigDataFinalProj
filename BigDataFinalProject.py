#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from folium.plugins import MeasureControl
import requests as rq
import json
import pyproj
from geopy import distance
import ast
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re


# https://opendata-ajuntament.barcelona.cat/en/
# http://insideairbnb.com/get-the-data/
# https://laura-an.carto.com/tables/shapefile_distrito_barcelona/public
# https://www.bcn.cat/tercerlloc/files/opendatabcn_pics-js.json
# https://download.geofabrik.de/europe/spain/cataluna.html
# https://raw.githubusercontent.com/martgnz/bcn-geodata/master/barris/barris.geojson

# In[2]:


df = pd.read_csv("barca_data_detailed.csv")
sum_df = pd.read_csv("barca_summer_data.csv")
type(df['price'][0])
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float) #Change prices to a float
sum_df['price'] = sum_df['price'].str.replace('$', '').str.replace(',', '').astype(float)
type(df['price'][0])


# In[3]:


#drop entries without a rating
df = df.loc[(df['review_scores_rating'].notnull()) & (df['review_scores_rating'] != 0)]
df = df.loc[(df['price'].notnull()) & (df['price'] != 0)]
df = df.loc[(df['neighbourhood_cleansed'].notnull()) & (df['neighbourhood_cleansed'] != 0)]
df = df.loc[(df['room_type'].notnull()) & (df['room_type'] != 0)]
df = df.loc[(df['beds'].notnull()) & (df['beds'] != 0)]
df = df.loc[(df['number_of_reviews'].notnull()) & (df['number_of_reviews'] != 0)]

sum_df = sum_df.loc[(sum_df['review_scores_rating'].notnull()) & (sum_df['review_scores_rating'] != 0)]
sum_df = sum_df.loc[(sum_df['price'].notnull()) & (sum_df['price'] != 0)]
sum_df = sum_df.loc[(sum_df['neighbourhood_cleansed'].notnull()) & (sum_df['neighbourhood_cleansed'] != 0)]
sum_df = sum_df.loc[(sum_df['room_type'].notnull()) & (sum_df['room_type'] != 0)]
sum_df = sum_df.loc[(sum_df['beds'].notnull()) & (sum_df['beds'] != 0)]
sum_df = sum_df.loc[(sum_df['number_of_reviews'].notnull()) & (sum_df['number_of_reviews'] != 0)]


# In[4]:


#df['latitude']


# In[5]:


#df['longitude']


# In[6]:


df['value'] = df['review_scores_rating']/df['price']
#df['value']


# geometry = gpd.points_from_xy(df.longitude, df.latitude)
# gdf = gpd.GeoDataFrame(df, geometry=geometry)
# gdf.crs = "EPSG:4326"
# #gdf.geometry

# In[8]:


#Create a map of Barcelona
barca_map = folium.Map(location=[41.385, 2.173], zoom_start=10)
barca_summer_map = folium.Map(location=[41.385, 2.173], zoom_start=10)

#Create a marker cluster to add Airbnb listings to the map
marker_cluster = MarkerCluster(name='Airbnbs').add_to(barca_map)
sum_marker_cluster = MarkerCluster(name='Airbnbs').add_to(barca_summer_map)


#Add markers for each Airbnb listing to the marker cluster
for lat, lon, price, name, neighborhood in zip(df['latitude'], df['longitude'], df['price'], df['name'], df['neighbourhood_cleansed']):
    folium.Marker(
        location=[lat, lon],
        popup=f'{neighborhood}\n\n{name}\n${price}/night',
        icon=None,
    ).add_to(marker_cluster)
    
for lat, lon, price, name, neighborhood in zip(sum_df['latitude'], sum_df['longitude'], sum_df['price'], sum_df['name'], sum_df['neighbourhood_cleansed']):
    folium.Marker(
        location=[lat, lon],
        popup=f'{neighborhood}\n\n{name}\n${price}/night',
        icon=None,
    ).add_to(sum_marker_cluster)


# In[9]:


#Get tourist attraction json data
response = rq.get("https://www.bcn.cat/tercerlloc/files/opendatabcn_pics-js.json")
tourist_json = response.json()


# In[10]:


#We need ro restructure the json so that it is in a GeoJSON format that folium can read it

src_crs = "EPSG:25831" #Coordinates the dataset came in
dst_crs = "EPSG:4326"  #Coordiantes we want
transformer = pyproj.Transformer.from_crs(src_crs, dst_crs)  #The pyproj library can convert different types of coords



features = []
for item in tourist_json:
    x = item['addresses'][0]['location']['geometries'][0]['coordinates'][0]
    y = item['addresses'][0]['location']['geometries'][0]['coordinates'][1]
    longitude, latitude = transformer.transform(x, y)
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [latitude, longitude]
        },
        "properties": {
            "name": item['name'],
            "neighborhood": item['addresses'][0]['neighborhood_name']
        }
    }
    features.append(feature)


# In[11]:


geojson_data = {
    "type": "FeatureCollection",
    "features": features
}

with open('barca_tourist.geojson', 'w') as f:
    json.dump(geojson_data, f)


# In[12]:


with open('barca_tourist.geojson', 'r') as f:
    barca_tourist_data = json.load(f)
    
icon = folium.Icon(color='orange', icon='info-sign')
tourism_marker = folium.Marker(popup='Attraction', icon=icon)

folium.GeoJson(barca_tourist_data, name="Tourist Attractions", marker=tourism_marker).add_to(barca_map)


# In[13]:


#Now get the neighborhood GeoJSON
response = rq.get("https://raw.githubusercontent.com/martgnz/bcn-geodata/master/barris/barris.geojson")
neighborhoods = response.json()

with open('barca_neighborhoods.geojson', 'w') as f:
    json.dump(neighborhoods, f)


# In[14]:


with open('barca_neighborhoods.geojson', 'r') as f:
    neighborhoods = json.load(f)


folium.GeoJson(neighborhoods, name="Neighborhoods").add_to(barca_map)
folium.GeoJson(neighborhoods, name="Neighborhoods").add_to(barca_summer_map)


# In[15]:


#Group Airbnb listings by neighborhood and calculate average price
neighborhood_prices = df.groupby('neighbourhood_cleansed').mean()['price']
sum_neighborhood_prices = sum_df.groupby('neighbourhood_cleansed').mean()['price']



#Create a choropleth map that color-codes the neighborhoods based on average price of Airbnbs in them
folium.Choropleth(
    geo_data=neighborhoods,
    name='Average Price',
    data=neighborhood_prices,
    key_on='feature.properties.NOM',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Average Price'
).add_to(barca_map)

folium.Choropleth(
    geo_data=neighborhoods,
    name='Average Price',
    data=sum_neighborhood_prices,
    key_on='feature.properties.NOM',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Average Price'
).add_to(barca_summer_map)


# In[16]:


#Group Airbnb listings by neighborhood and calculate average rating
neighborhood_ratings = df.groupby('neighbourhood_cleansed').mean()['review_scores_rating']
sum_neighborhood_ratings = sum_df.groupby('neighbourhood_cleansed').mean()['review_scores_rating']



#Create a choropleth map that color-codes the neighborhoods based on average rating of Airbnbs in them
folium.Choropleth(
    geo_data=neighborhoods,
    name='Average Rating',
    data=neighborhood_ratings,
    key_on='feature.properties.NOM',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Average Ratings'
).add_to(barca_map)

folium.Choropleth(
    geo_data=neighborhoods,
    name='Average Rating',
    data=sum_neighborhood_ratings,
    key_on='feature.properties.NOM',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Average Ratings'
).add_to(barca_summer_map)


# In[17]:


#Group Airbnb listings by neighborhood and calculate average location rating
neighborhood_location_ratings = df.groupby('neighbourhood_cleansed').mean()['review_scores_location']
sum_neighborhood_location_ratings = sum_df.groupby('neighbourhood_cleansed').mean()['review_scores_location']


#Create a choropleth map that color-codes the markers based on average location rating of Airbnbs in them
folium.Choropleth(
    geo_data=neighborhoods,
    name='Average Location Rating',
    data=neighborhood_location_ratings,
    key_on='feature.properties.NOM',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Average Location Ratings'
).add_to(barca_map)

folium.Choropleth(
    geo_data=neighborhoods,
    name='Average Location Rating',
    data=sum_neighborhood_location_ratings,
    key_on='feature.properties.NOM',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Average Location Ratings'
).add_to(barca_summer_map)


# In[18]:


folium.LayerControl().add_to(barca_map)
MeasureControl(position='topleft', active_color='blue', completed_color='green').add_to(barca_map)

folium.LayerControl().add_to(barca_summer_map)
MeasureControl(position='topleft', active_color='blue', completed_color='green').add_to(barca_summer_map)


# In[19]:


barca_map


# In[22]:


barca_map.save('barca_map.html')


# In[20]:


barca_summer_map


# In[23]:


barca_summer_map.save('barca_summer_map.html')


# In[ ]:


barca_shapefile = gpd.read_file('barcelona.shp')


# fig, ax = plt.subplots(figsize=(10,10))
# barca.plot(ax=ax, color='white', edgecolor='black')
# gdf.plot(ax=ax, markersize=5, color='red', alpha=0.5)
# plt.title('Airbnb rental properties in the city')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()

# In[ ]:


geometry = gpd.points_from_xy(df.longitude, df.latitude)
gdf = gpd.GeoDataFrame(df, geometry=geometry)
gdf.crs = "EPSG:4326"


# In[ ]:


barca_neighborhood = gpd.sjoin(gdf, barca_shapefile, predicate='within')


# In[ ]:


price_breakdown = barca_neighborhood.groupby('neighbourhood_cleansed').agg({
    'price': ['mean', 'median'],
    'geometry': 'first'
})


# In[ ]:


fig, ax = plt.subplots(figsize=(24,12))
ax.plot(price_breakdown['price'])
plt.title('Neighborhood Prices', fontsize=12)
plt.xlabel('Neighborhood', fontsize=6)
plt.tick_params(width=1, labelsize=5)
ax.set_xlim(price_breakdown.index.min(), price_breakdown.index.max())
plt.show()


# In[ ]:


sorted_prices = price_breakdown.sort_values(by=('price', 'mean'), ascending=False)
print(sorted_prices.to_string())


# In[ ]:


value_breakdown = barca_neighborhood.groupby('neighbourhood_cleansed').agg({
    'value': ['mean', 'median'],
    'geometry': 'first'
})


# In[ ]:


sorted_values = value_breakdown.sort_values(by=('value', 'mean'), ascending=False)
print(sorted_values.to_string())


# In[ ]:


fig, ax = plt.subplots(figsize=(24,12))
ax.plot(value_breakdown['value'])
plt.title('Neighborhood Values', fontsize=12)
plt.xlabel('Neighborhood', fontsize=6)
plt.tick_params(width=1, labelsize=5)
ax.set_xlim(value_breakdown.index.min(), value_breakdown.index.max())
plt.show()


# In[ ]:


#print(plt.style.available)


# In[ ]:





# In[ ]:


#barca_tourist_data['features'][0]['geometry']['coordinates']


# In[ ]:


#Note: This takes a while to run (15-20 mins on my computer)

distances = []
for i, airbnb in df.iterrows():
    min_distance = 99999
    point1 = (airbnb['latitude'], airbnb['longitude'])
    for attraction in barca_tourist_data['features']:
        tour_lat = attraction['geometry']['coordinates'][1]
        tour_lon = attraction['geometry']['coordinates'][0]
        point2 = (float(tour_lat), float(tour_lon))
        #print(point1)
        dis = distance.distance(point1, point2).miles
        if dis < min_distance:
            min_distance = dis
    distances.append(min_distance)
df['distance'] = distances


# In[ ]:


with open('distances_tourism.txt', 'w') as f:
    for distance in distances:
        f.write(str(distance)+'\n')


# print(distances)

# In[ ]:


#print(df['price'].max())


# In[ ]:


with open('distances_tourism.txt', 'r') as f:
    distances = f.readlines()

distances = [distance.strip() for distance in distances]
df['distance'] = distances


# In[ ]:


#df['distance']


# In[ ]:


plt.scatter(df['distance'], df['price'])
plt.xlabel('Distance (miles)')
plt.ylabel('Price ($)')
plt.ylim(0, 300)
plt.xlim(0, 10)
#plt.set_xticklabels([])
plt.title('Airbnb price vs. Distance to Nearest Tourist Attraction')
plt.show()


# In[ ]:


num_amenities = []
max_amens = 0
i_max = 0
for i, row in df.iterrows():
    num = 0
    as_list = ast.literal_eval(row['amenities'])
    for amen in as_list:
        num+=1
    num_amenities.append(num)
    if num > max_amens:
        i_max = i
    
df['num_amenities'] = num_amenities


# In[ ]:


plt.scatter(df['num_amenities'], df['price'], s=6)

plt.xlabel('Number of Amenities')
plt.ylabel('Price')
plt.ylim(0, 600)
plt.xlim(0, 80)

plt.show()


# In[ ]:


df['amenities'][i_max]


# In[ ]:


plt.scatter(df['beds'], df['price'], s=15)

plt.xlabel('Number of Beds')
plt.ylabel('Price')
plt.ylim(0, 800)
plt.xlim(0, 20)

plt.show()


# In[ ]:


df_private_homes = df[df['room_type'] == 'Private room']

plt.scatter(df_private_homes['beds'], df_private_homes['price'])
plt.xlabel('Number of Beds')
plt.ylabel('Price')
plt.ylim(0, 1000)
plt.title('Private Homes Only')
plt.show()


# In[ ]:


stop_words = set(stopwords.words('english'))
#Loop through each row in the description column
for i, row in df.iterrows():
    #Convert the text to lowercase and tokenize it
    df.loc[i, 'description'] = str(df.loc[i, 'description'])
    text = df.loc[i, 'description'].lower()
    words = word_tokenize(text)
    #Remove stop words from the text and join the remaining words back into a single string
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
    filtered_text = ' '.join(filtered_words)
    #Update the 'description' column with the filtered text
    df.loc[i, 'description'] = filtered_text


# In[ ]:


#df['description']


# In[ ]:


#Concatenate the text into one string
text = "".join(df['description'])


#print(text[:10000])


# In[ ]:


#Remove commonly occuring stuff that isn't words
text = re.sub(r'[^\w\s]+', '', text)
text = text.replace(' br ', '')
text = text.replace(' b ', '')


# In[ ]:


#print(text[:10000])


# In[ ]:


#Make a dict of the word frequencies
word_dict = Counter(text.split())


# In[ ]:


#Make the wordcloud
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', min_font_size = 10, regexp=r'\w+').generate_from_frequencies(word_dict)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[ ]:




