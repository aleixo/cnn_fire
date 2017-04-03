# coding=utf-8
#https://developers.google.com/drive/v3/web/quickstart/python
from __future__ import print_function
import httplib2
import os
import io
from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools
from apiclient.http import MediaIoBaseDownload
from apiclient.http import MediaFileUpload
import sys
import argparse
from pyfcm import FCMNotification

"""
DESCRIPTION
Script with class that manages operations with Google. Send file, uploads file and send push notification

import sys
sys.path.insert(0,'utils/google')
import GoogleDrive_manager
"""

class GoogleManager:

    def __init__(self):
        self.SCOPES = 'https://www.googleapis.com/auth/drive'
        self.CLIENT_SECRET_FILE = 'GoogleDrive_Client_secret.json'
        self.APPLICATION_NAME = 'pythonscript'

    def init_for_upload(self,upload_file=None,upload_file_name=None):
        if upload_file and upload_file_name:
            self.upload_manager(upload_file,upload_file_name)
        else:
            raise ValueError("[ERROR] Object initializer has to have file name to upload and name of uploaded file in upload mode. Initialize object with mode, upload filename and upload destination name")               

    def init_for_download(self,download_file=None):
        if download_file:                
            self.download_manager(download_file)
        else:
            raise ValueError("[ERROR] Object initializer has to have file name to download in download mode. Initialize object with mode and file name to download") 

    def init_for_list(self,):
        self.download_manager(True)

    def download_file(self,file_id, mimeType, filename,drive_service):
        if "google-apps" in mimeType:
            return
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.FileIO(filename, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("[PROGRESS]  Download %d%%." % int(status.progress() * 100))

    def get_credentials(self):
        SCOPES = 'https://www.googleapis.com/auth/drive'
        CLIENT_SECRET_FILE = 'GoogleDrive_Client_secret.json'
        APPLICATION_NAME = 'pythonscript'
        home_dir = os.path.expanduser('~')
        credential_dir = os.path.join(home_dir, '.credentials')
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir)
        credential_path = os.path.join(credential_dir,'drive-python-quickstart.json')
        store = oauth2client.file.Storage(credential_path)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
            flow.user_agent = APPLICATION_NAME
            if flags:
                credentials = tools.run_flow(flow, store, flags)
            else:
                credentials = tools.run(flow, store)        
        return credentials

    def upload_manager(self,fileToUpload,nameToUpload):
        credentials = self.get_credentials()
        http = credentials.authorize(httplib2.Http())
        drive_service = discovery.build('drive', 'v3', http=http)

        file_metadata = {'name' : nameToUpload,'mimeType' : 'application/octet-stream'}
        media = MediaFileUpload(fileToUpload,mimetype='application/octet-stream',resumable=True)
        file = drive_service.files().create(body=file_metadata,media_body=media,fields='id').execute()
        print("[INFO] File with name {} uploaded to Google Drive".format(nameToUpload))

    def download_manager(self,fileToDownload=None,list = False):                   
        credentials = self.get_credentials()
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('drive', 'v3', http=http)
        results = service.files().list(pageSize=10,fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            print('No files found.')
        else:
            for item in items:    
                name = str(item["name"].encode('ascii', 'ignore'))        
                print(name)            
                if name == fileToDownload and not list:
                    credentials = self.get_credentials()
                    http = credentials.authorize(httplib2.Http())
                    drive_service = discovery.build('drive', 'v3', http=http)
                    self.download_file(item['id'],"text/plain",item['name'],drive_service)      

    def sendPushForFire(self,percentage):
        message_body = self.pushFireMessageBody(percentage)
        if message_body is not "":
            push_service = FCMNotification(api_key="AIzaSyCav3GxdR91EKG2Z3TQNrEtP3F34jTy9oY")
            registration_id = "flafqcTKoYM:APA91bGildgQPgZQ0ss7DavoWhAHXZmMI8cK47c5eX28CAVJD_lZiJ9DLadCCMqYFKtAHCQZG6gnvCgSrDC9BLHOgW5dR_14zPy4ZedFQT2kqzGI-kX4H0wK2LY4TURRxU2WT43a037S"
            message_title = "Fire alarm system"            
            result = push_service.notify_single_device(registration_id=registration_id, message_title=message_title, message_body=message_body,sound="Default",badge=1)
            return result
        else:
            return "sem alarme"

    def sendPushForEvent(self,message_body):    
        if message_body is not "":
            push_service = FCMNotification(api_key="AIzaSyCav3GxdR91EKG2Z3TQNrEtP3F34jTy9oY")
            registration_id = "flafqcTKoYM:APA91bGildgQPgZQ0ss7DavoWhAHXZmMI8cK47c5eX28CAVJD_lZiJ9DLadCCMqYFKtAHCQZG6gnvCgSrDC9BLHOgW5dR_14zPy4ZedFQT2kqzGI-kX4H0wK2LY4TURRxU2WT43a037S"
            message_title = "Events system"
            message_body = "[EVENT] "+message_body
            result = push_service.notify_single_device(registration_id=registration_id, message_title=message_title, message_body=message_body,sound="Default",badge=1)
            return result
        else:
            return "sem alarme"

    def pushFireMessageBody(self,percentage):
        message_body = ""
        if percentage > 70 and percentage < 80:
            message_body = "HIGH - Percentage of certainty above 70%"
        elif percentage > 80 and percentage <90:
            message_body = "URGENT - Percentage of certainty above 80%"
        elif percentage > 90:
            message_body = "CRITICAL - Percentage of certainty above 90%"
        return "[FIRE ALARM] "+message_body

    def sendPush(message_title,percentage):
        message_body = managePush(percentage)
        if message_body is not "":
            push_service = FCMNotification(api_key="AIzaSyCav3GxdR91EKG2Z3TQNrEtP3F34jTy9oY")
            registration_id = "flafqcTKoYM:APA91bGildgQPgZQ0ss7DavoWhAHXZmMI8cK47c5eX28CAVJD_lZiJ9DLadCCMqYFKtAHCQZG6gnvCgSrDC9BLHOgW5dR_14zPy4ZedFQT2kqzGI-kX4H0wK2LY4TURRxU2WT43a037S"
            message_title = message_title
            message_body = message_body
            result = push_service.notify_single_device(registration_id=registration_id, message_title=message_title, message_body=message_body,sound="Default",badge=1)
            return result
        else:
            return "sem alarme"


    def managePush(percentage):
        message_body = ""
        if percentage > 40 and percentage < 60:
            message_body = "Alarme entre 40 a 60"
        elif percentage > 60 and percentage <80:
            message_body = "Alarme entre 60 a 80"
        elif percentage > 100:
            message_body = "Alarme entre 80 a 100"
        return message_body
drive = GoogleManager()
#drive.init_for_list()
#drive.init_for_download("teste.json")
#drive.init_for_upload("r.json","teste.json")


