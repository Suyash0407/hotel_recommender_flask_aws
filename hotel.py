"""
@author: Suyash
"""

# =============================================================================
# # LIBRARIES
from flask import Flask, render_template, request
import pandas as pd
import numpy as np 
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from tabulate import tabulate
nltk.download('punkt')
# =============================================================================

# =============================================================================
# # IMPORTING DATA 

hotel_details = pd.read_csv('Hotel_details.csv')

hotel_room_attr = pd.read_csv('Hotel_Room_attributes.csv')

hotels_room_price = pd.read_csv('hotels_RoomPrice.csv')
# =============================================================================

# =============================================================================
# # DATA PRE-PROCESSING

# For hotel room attributes
hotel_room_attr.isnull().sum()
hotel_room_attr.shape
hotel_room_attr.columns
hotel_room_attr_na=hotel_room_attr.dropna()

# dropping id as hotelcode is a unique column for hotel room attributes
hotel_room_attr.drop(['id'], axis=1, inplace=True)

# This shows that duplicates exists 
cnt_hotel = hotel_room_attr['hotelcode'].value_counts()
df1 = hotel_room_attr[hotel_room_attr['hotelcode']== 5344]

# Remove Duplicates based on hotel code and room type 
hotel_room_attr.drop_duplicates(keep='first', inplace = True,subset=['hotelcode','roomtype'])

# Dropping na values as the prices data for NA values is also not there.
hotel_room_attr_na=hotel_room_attr.dropna()

# Joining both the tables ie room attr and hotel details

df_hotel = pd.merge(hotel_room_attr_na,hotel_details,left_on='hotelcode',right_on='hotelid',how='inner')

df_hotel.isnull().sum()

# Dropping those columns which I will not use in recommendation
df_hotel.columns
df_hotel.drop(['address','zipcode','url','curr',
               'latitude','longitude','Source'], axis=1, inplace=True)

#Calculate Number of guest based on the room type 

ary_hotel = df_hotel['roomtype'].value_counts()

# Function for deleting all the punctuations from a column
def remove_punctuations(df,column):
    lst = []
    for line in df[column]:
        trans = line.maketrans('','',string.punctuation)
        line = line.translate(trans)
        lst.append(line.lower())
    df.drop([column], axis=1, inplace=True)
    df[column] = lst


# Deleting all punctuations from the room type
remove_punctuations(df_hotel,'roomtype')

# For hotel Room Price
 
# Removing below columns as room price can be uniquely identified by hotelcode 
hotels_room_price.drop(['id', 'refid'], axis=1, inplace=True)

# Dropping Duplicate values 
cnt_hotel = hotels_room_price['hotelcode'].value_counts()


# Dropping few columns 
hotels_room_price.drop(['websitecode','dtcollected', 'ratedate','los',
                        'guests','ratetype','sourceurl','roomamenities','ispromo',
                        'closed', 'discount', 'promoname', 'status_code', 
                        'taxstatus','taxtype', 'taxamount', 'proxyused', 
                        'israteperstay','mealinclusiontype', 'hotelblock', 
                        'input_dtcollected'], axis=1, inplace=True)
del hotels_room_price['ratedescription']
del hotels_room_price['netrate']
del hotels_room_price['maxoccupancy']

# For few of the the room types we dont have price 
# So just replacing that price with nan values
for i in range(0,hotels_room_price.shape[0]):
    if hotels_room_price.loc[i,'onsiterate'] == 0:
        hotels_room_price.loc[i,'onsiterate'] = np.nan

# Dropping nan values
hotel_room_price_na =  hotels_room_price.dropna()


# Since in the data onsite rate was varying based on max occupancy but neglecting that
# and considering only higher side price for each room type to cover all case
# and to be on a safer side
hotel_room_price_na.sort_values(by=['onsiterate'],ascending=False,inplace=True)
hotel_cost=hotel_room_price_na.drop_duplicates(subset=['hotelcode','roomtype'],keep='first')

# Deleting all punctuations from the room type
room_type = []
for line in hotel_cost['roomtype']:
    trans = line.maketrans('','',string.punctuation)
    line = line.translate(trans)
    room_type.append(line.lower())

hotel_cost.drop(['roomtype'], axis=1, inplace=True)
hotel_cost['roomtype'] = room_type

hotel_cost.drop(['roomtype'], axis=1, inplace=True)
hotel_cost['roomtype'] = room_type

# Some formatting problem with hotelcode
hotel_code = []
for line in hotel_cost['hotelcode']:
    line = int(line)
    hotel_code.append(line)

hotel_cost.drop(['hotelcode'], axis=1, inplace=True)
hotel_cost['hotelcode'] = hotel_code

# Joining Both the data frame ie df_hotel and hotel_cost
hotel_data=pd.merge(df_hotel,hotel_cost,
                left_on=['hotelcode','roomtype'],
                right_on=['hotelcode','roomtype'],
                how='inner')

#Calculating number of guests for each type of room 

room_no=[
     ('king',2),
   ('queen',2), 
    ('triple',3),
    ('master',3),
   ('family',4),
   ('murphy',2),
   ('quad',4),
   ('double-double',4),
   ('mini',2),
   ('studio',1),
    ('junior',2),
   ('apartment',4),
    ('double',2),
   ('twin',2),
   ('double-twin',4),
   ('single',1),
     ('diabled',1),
   ('accessible',1),
    ('suite',2),
    ('one',2)
   ]

def calc1():
    guests_no=[]
    for i in range(hotel_data.shape[0]):
        temp=hotel_data['roomtype'][i].lower().split()
        flag=0
        for j in range(len(temp)):
            for k in range(len(room_no)):
                if temp[j]==room_no[k][0]:
                    guests_no.append(room_no[k][1])
                    flag=1
                    break
            if flag==1:
                break
        if flag==0:
            guests_no.append(2)
    hotel_data['guests_no']=guests_no

calc1()

# remove punctuations from room amenities      
remove_punctuations(hotel_data,'roomamenities')
    
# Stemming of room amenities column for consistant rows  
porter=PorterStemmer()  
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

 
 
stemmed_sentences = []
for line in hotel_data['roomamenities']:
    sentence = stemSentence(line)
    stemmed_sentences.append(sentence)

hotel_data['stemmed_roomamenities'] = stemmed_sentences
# =============================================================================

# =============================================================================
## Function for recommender

def hotel_recommender(country,city,guest_number,room_amenities,number_of_hotels):

    #Countary filter 
    country = country.lower()
    df_country_filter = hotel_data[hotel_data['country'].str.lower() == country]
    
    #City Filter 
    city = city.lower()
    df_city_filter = df_country_filter[df_country_filter['city'].str.lower() == city]
    
    # Guest Number filter:
    
    df_guest_number_filter = df_city_filter[df_city_filter['guests_no'] == guest_number]
    
    # Room Aminities 
    
    # Calculating scrore on basis of max aminities present
    room_amenities = 'air conditioning,tv,tea'
    room_amenities = room_amenities.split(',')
    room_amenities_new = []
    for i in room_amenities:
        c = stemSentence(i)
        room_amenities_new.append(c)
    
    
    amenities_score = []
    for line in df_guest_number_filter['stemmed_roomamenities']:
        lst = []
        for i in room_amenities_new:
            a = re.findall(i,line) 
            lst +=a
        amenities_score.append(len(lst))
    
    df_guest_number_filter['amenities_score'] = amenities_score
    
    df_all_filters_applied = df_guest_number_filter
    
    df_all_filters_applied.columns
    df_all_filters_applied.sort_values(by=['starrating','amenities_score'],ascending=False,inplace=True)
    
    # For number of hotels
    final = df_all_filters_applied[['hotelname','city','country','guests_no','roomtype','onsiterate']].head(number_of_hotels)
    final = tabulate(final, tablefmt="pipe", headers="keys")
    return final 

print('\n\n\t\tStarting the server please wait')  
# =============================================================================

# =============================================================================
# # FLASK CODE

app = Flask(__name__)

@app.route('/home', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        country = request.form.get('country')
        city = request.form.get('city')
        guest_number = int(request.form.get('guest_number'))
        room_amenities = request.form.get('room_amenities')
        number_of_hotels = int(request.form.get('number_of_hotels'))
        hotel_obj = hotel_recommender(country,city,guest_number,room_amenities,number_of_hotels)
        hotel = hotel_obj
        return render_template('index_hotel.html', hotellist = hotel)
    return render_template('index_hotel.html', hotellist = 'No data')

if __name__ == '__main__':
   app.run(debug = True)
# =============================================================================
