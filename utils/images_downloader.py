#coding:utf-8
import json
from skimage import io
import cv2
import argparse
import os
import random
from utils import  Utils
from google_images_downloader import GoogleDownloader
import subprocess
from imutils import paths
import os
from bs4 import BeautifulSoup
from urllib2 import urlopen
from urllib2 import Request
from urllib2 import HTTPError
from urllib2 import build_opener

"""
QUERIES:
forrest
mountain
amazon forrest
green forrest
forrest sunset

wildfire
forrest fire
forrest burning
burning summer
forrest firefighters
forrest smoke

DESCRIPTION
Script to download images from one API and one site

USAGE:
python images_downloader.py -q forrest -d images/fire/  -p forrest -r 128
"""

ap = argparse.ArgumentParser()
ap.add_argument("-q","--query",required=True,help="Image search query.")
ap.add_argument("-d","--destination_dir",required=True,help="Images destination dir")
ap.add_argument("-p","--prefix",required=True,help="Sets the image prefix, category of the image, and top directory")
ap.add_argument("-r","--resize_dim",required=True,help="The squared dimension of the new image")
args = vars(ap.parse_args())

new_dim = args['resize_dim']
destination_dir = args['destination_dir']

query = urlopen("https://pixabay.com/api/?key=3069392-4e571367e1943bff61c178bc0&q="+args['query']).read().decode('utf8')
pages = 0

#Os nomes não podem ter espaços
#Normalizar as dimensões da imagem
#Forçar a função a correr outra vez ..... Porque? Ver isso.
		
def getPixabayImages():
	print("[INFO] Pixabay images")

	for key,value in json.loads(query).items():
		if key == "totalHits":					
			if value < 200:			
				pages = 1
			else:
				pages = int(value / 200) + 1			

	print("--------------------------------")
	print("[INFO] Num pages {}".format(pages))
	print("--------------------------------")

	num_images = 0
	for p in range(pages):
		print("--------------------------------")
		print("[INFO] Getting page {}".format(p))
		print("--------------------------------")
		url = "https://pixabay.com/api/?key=3069392-4e571367e1943bff61c178bc0&q="+args['query']+"&per_page=200&pages&page="+str(p + 1)

		print(url)

		data = urlopen(url).read().decode('utf-8')

		for key,value in json.loads(data).items():
			if key == "hits":	
				for i,hit in enumerate(value):
					num_images += 1
					try:
						image = io.imread(hit["webformatURL"])
						image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)						
						path = os.path.join(destination_dir+"/",args['prefix']+"/"+args['query']+"_"+str(random.uniform(0.0,10000.0))+".jpg")
						cv2.imwrite(path,cv2.resize(image,(int(new_dim),int(new_dim)),interpolation = cv2.INTER_AREA))
						print("[INFO] Downloaded image num {} and saved to {}".format(i,path))						
					except Exception as e:
						print("ERROR {}".format(e))

	#utilMethods = Utils()
	#utilMethods.deleteDuplicated2(dir=destination_dir)



def getSiteImgFromTag():
	print("[INFO] Forestry images")	
	try:
		site="http://www.forestryimages.org/search/action.cfm?q="+args['query']+"&start=1"		
		req = Request(site, headers={'User-Agent' : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.30 (KHTML, like Gecko) Ubuntu/11.04 Chromium/12.0.742.112 Chrome/12.0.742.112 Safari/534.30"})
		page = urlopen(req)
		soup = BeautifulSoup(page,"lxml")
	except HTTPError as e:
		print("ERROR {}".format(e))

	i_to_break = 0
	num_images = 0
	for i,divTag in enumerate(soup.findAll('strong')):			
		#ONE EXTRA STRON TAG
		if divTag.text == "Your search term exactly matched the following items:":
			i_to_break = 1		
		if  i > i_to_break:
			
			num_images = divTag.text
			num_images = int(num_images.replace(",",""))

			num_pages = (int)(num_images / 200)
			print("[INFO] Total of {} images in {} pages".format(num_images,num_pages))			
			break

	next_page = 1	
	for i in range(num_pages + 1):
		print("--------------------------------")
		print("[INFO] Getting page {}".format(i))
		print("--------------------------------")
		try:
			site="http://www.forestryimages.org/search/action.cfm?q="+args['query']+"&start=1"		
			req = Request(site, headers={'User-Agent' : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.30 (KHTML, like Gecko) Ubuntu/11.04 Chromium/12.0.742.112 Chrome/12.0.742.112 Safari/534.30"})
			page = urlopen(req)
			soup = BeautifulSoup(page,"lxml")
		
		except HTTPError as e:
			print("ERROR {}".format(e))
		for imgNum,imgTag in enumerate(soup.findAll('img')):
			try:
				imgUrl = str(imgTag['src'])
				if imgUrl.startswith('https://'):
					image = io.imread(imgUrl)
					imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					path = os.path.join(destination_dir + "/", args['prefix'] + "/" + args['query'] + "_" + str(
						random.uniform(0.0, 10000.0)) + ".jpg")
					cv2.imwrite(path,cv2.resize(imageRGB,(int(new_dim),int(new_dim)),interpolation = cv2.INTER_AREA))	
					print("[INFO] Downloaded image num {} from page {} and saved to {}".format(imgNum,i,path))		
			except Exception as e:
				print("ERROR {}".format(e))
		next_page += 200

google_Downloader = GoogleDownloader()
#google_Downloader.startDownload(args['query'],args['prefix'])
#getPixabayImages()
getSiteImgFromTag()

utilMethods = Utils()
#utilMethods.deleteDuplicated(destination_dir)

