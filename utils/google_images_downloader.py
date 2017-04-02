#https://github.com/hardikvasa/google-images-download/blob/master/google-images-download.py

import time
import sys    #Importing the System Library
from urllib2 import urlopen
from urllib2 import Request
from urllib2 import HTTPError
from urllib2 import URLError
import random

class GoogleDownloader(object):

    search_keyword = ['wild fire', "mountain fire", "forrest fire"]
    keywords = [' high resolution']

    def download_page(self,url):

        version = (3,0)
        cur_version = sys.version_info

        if cur_version >= version:
            import urllib.request
            try:
                headers = {}
                headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
                req = urllib.request.Request(url, headers = headers)
                resp = urllib.request.urlopen(req)
                respData = str(resp.read())
                return respData
            except Exception as e:
                print(str(e))
        else:                        #If the Current Version of Python is 2.x
            import urllib2
            try:
                headers = {}
                headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
                req = urllib2.Request(url, headers = headers)
                response = urllib2.urlopen(req)
                page = response.read()
                return page
            except:
                return"Page Not found"

    def _images_get_next_item(self,s):
        start_line = s.find('rg_di')
        if start_line == -1:    #If no links are found then give an error!
            end_quote = 0
            link = "no_links"
            return link, end_quote
        else:
            start_line = s.find('"class="rg_meta"')
            start_content = s.find('"ou"',start_line+1)
            end_content = s.find(',"ow"',start_content+1)
            content_raw = str(s[start_content+6:end_content-1])
            return content_raw, end_content

    def _images_get_all_items(self,page):
        items = []
        while True:
            item, end_content = self._images_get_next_item(page)
            if item == "no_links":
                break
            else:
                items.append(item)      #Append all the links in the list named 'Links'
                time.sleep(0.1)        #Timer could be used to slow down the request for image downloads
                page = page[end_content:]
        return items

    def startDownload(self,querie,folder):

        self.search_keyword = [querie]
        t0 = time.time()


        i= 0
        while i<len(self.search_keyword):
            items = []
            iteration = "Item no.: " + str(i+1) + " -->" + " Item name = " + str(self.search_keyword[i])
            print (iteration)
            print ("Evaluating...")
            search_keywords = self.search_keyword[i]
            search = search_keywords.replace(' ','%20')
            j = 0
            while j<len(self.keywords):
                pure_keyword = self.keywords[j].replace(' ','%20')
                url = 'https://www.google.com/search?q=' + search + pure_keyword + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
                raw_html =  (self.download_page(url))
                time.sleep(0.1)
                items = items + (self._images_get_all_items(raw_html))
                j = j + 1

            print ("Total Image Links = "+str(len(items)))
            print ("\n")
            i = i+1



            info = open('output.txt', 'a')
            info.write(str(i) + ': ' + str(self.search_keyword[i-1]) + ": " + str(items) + "\n\n\n")
            info.close()

        t1 = time.time()
        total_time = t1-t0
        print("Total time taken: "+str(total_time)+" Seconds")
        print ("Starting Download...")

        k=0
        errorCount=0
        while(k<len(items)):

            try:
                req = Request(items[k], headers={"User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
                response = urlopen(req)

                output_file = open("Downloads/"+folder+"/"+self.search_keyword[0]+"_"+str(random.uniform(0.0,10000.0))+".jpg",'wb')
                print("[GOOGLE IMAGED DOWNLOADER] Saved image "+str(k+1)+" to Downloads/"+folder+"/"+self.search_keyword[0]+"_"+str(random.uniform(0.0,10000.0))+".jpg")
                data = response.read()
                output_file.write(data)
                response.close();

                k=k+1;

            except IOError:   #If there is any IOError

                errorCount+=1
                print("IOError on image "+str(k+1))
                k=k+1;

            except HTTPError as e:  #If there is any HTTPError

                errorCount+=1
                print("HTTPError"+str(k))
                k=k+1;
            except URLError as e:

                errorCount+=1
                print("URLError "+str(k))
                k=k+1;

        print("\n")
        print("All are downloaded")
        print("\n"+str(errorCount)+" ----> total Errors")