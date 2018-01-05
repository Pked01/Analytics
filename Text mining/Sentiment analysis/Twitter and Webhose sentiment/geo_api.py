import requests
from IPython import embed
import datetime
import json
from requests.auth import HTTPBasicAuth

class geoAPI:
    def __init__(self):
        # self.base_url = "http://localhost:3000/api/geocode/"
        self.base_url = "https://opendata-maps.herokuapp.com/api/geocode/"


    def get_results(self, endpoint="", attribute_string=""):
        self.url_with_attr = self.base_url+endpoint+attribute_string
        r = requests.get(self.url_with_attr, auth=HTTPBasicAuth('geocode', 'ZXasqw12'))
        if r.status_code == 200:
            if r.text == "":
                return "nil"
            else:
                return r.json()
        else:
            return r

    #######################################   MAPS BASIC APIS    #######################################
    #   http://localhost:3000/api/geocode/poi_list
    def poi_list(self):
        endpoint = "poi_list"
        attribute_string = ""

        r = self.get_results(endpoint, attribute_string)
        return r

    #   http://localhost:3000/api/geocode/poi_list
    def country_list(self):
        endpoint = "country_list"
        attribute_string = ""

        r = self.get_results(endpoint, attribute_string)
        return r

    #   http://localhost:3000/api/geocode/poi_list
    def country_codes(self):
        endpoint = "country_codes"
        attribute_string = ""

        r = self.get_results(endpoint, attribute_string)
        return r


    #   http://localhost:3000/api/geocode/cities_by_country?country_name=India
    def cities_by_country(self, country_name=""):
        endpoint = "cities_by_country"
        attribute_string = "?country_name="+country_name

        r = self.get_results(endpoint, attribute_string)
        return r


    #######################################   MAPS APIS    #######################################


    #   http://localhost:3000/api/geocode/search?keyword=India
    def search(self, keyword=""):
        endpoint = "search"
        attribute_string = "?keyword="+keyword

        r = self.get_results(endpoint, attribute_string)
        return r

    #   http://localhost:3000/api/geocode/osm_search?keyword=hyderabad&country=india
    def osm_search(self, keyword="", country=""):
        endpoint = "osm_search"
        attribute_string = "?keyword="+keyword+"&country="+country

        r = self.get_results(endpoint, attribute_string)
        return r

    #   http://localhost:3000/api/geocode/reverse_search?lat=35.5087008&lng=97.39535869999999
    def reverse_search(self, lat="", lng=""):
        endpoint = "reverse_search"
        attribute_string = "?lat="+lat+"&lng="+lng

        r = self.get_results(endpoint, attribute_string)
        return r

    #   http://localhost:3000/api/geocode/pois?lat=20.593684&lng=78.96288
    def pois(self, lat="", lng="", radius="", type="", keyword=""):
        endpoint = "pois"
        attribute_string = "?lat="+lat+"&lng="+lng+"&radius="+radius+"&type="+type+"&keyword="+keyword

        r = self.get_results(endpoint, attribute_string)
        return r

    #   http://localhost:3000/api/geocode/osm_place_details?id=ChIJS05eKFhm0zsR1dQFG3Yreww
    def place_details(self, id=""):
        endpoint = "place_details"
        attribute_string = "?id="+id

        r = self.get_results(endpoint, attribute_string)
        return r

    #   http://localhost:3000/api/geocode/place_details?id=ChIJS05eKFhm0zsR1dQFG3Yreww
    def osm_place_details(self, id=""):
        endpoint = "osm_place_details"
        attribute_string = "?id="+id

        r = self.get_results(endpoint, attribute_string)
        return r


    #   http://localhost:3000/api/geocode/auto_complete?keyword=hyde
    def auto_complete(self, keyword=""):
        endpoint = "auto_complete"
        attribute_string = "?keyword="+keyword

        r = self.get_results(endpoint, attribute_string)
        return r

    #   http://localhost:3000/api/geocode/osm_auto_complete?keyword=hyde
    def osm_auto_complete(self, keyword=""):
        endpoint = "osm_auto_complete"
        attribute_string = "?keyword="+keyword

        r = self.get_results(endpoint, attribute_string)
        return r


    #######################################   HERE MAPS APIS    #######################################

    # Traffic Flow
    #   http://localhost:3000/api/geocode/traffic_flow?lat=17.450954&lng=78.380411
    def traffic_flow(self, lat=17.450954, lng=78.380411, zoom_level=16):
        endpoint = "traffic_flow"
        attribute_string = "?lat="+str(lat)+"&lng="+str(lng)+"&zoom_level="+str(zoom_level)

        r = self.get_results(endpoint, attribute_string)
        return r

    #   http://localhost:3000/api/geocode/traffic_flow_bbox?point1="17.6078088,78.6561694"&point2="17.2168886,78.1599217"
    def traffic_flow_bbox(self, point1="",point2=""):
        endpoint = "traffic_flow_bbox"
        attribute_string = "?point1="+str(point1)+"&point2="+str(point2)

        r = self.get_results(endpoint, attribute_string)
        return r

    # Routes
    #   http://localhost:3000/api/geocode/get_route?waypoint0=17.451241,78.381141&waypoint1=17.406355,78.402905&mode=fastest,car
    def get_route(self, waypoint0="17.451241,78.381141",waypoint1="17.406355,78.402905", mode="fastest,car"):
        endpoint = "get_route"
        attribute_string = "?waypoint0="+str(waypoint0)+"&waypoint1="+str(waypoint1)+"&mode="+str(mode)

        r = self.get_results(endpoint, attribute_string)
        return r

    #   http://localhost:3000/api/geocode/isoline?lat=17.450954&lng=78.380411
    def isoline(self, start="17.451241,78.381141",range="", rangetype="", mode=""):
        endpoint = "isoline"
        attribute_string = "?start="+str(start)+"&range="+str(range)+"&rangetype="+str(rangetype)+"&mode="+str(mode)

        r = self.get_results(endpoint, attribute_string)
        return r
