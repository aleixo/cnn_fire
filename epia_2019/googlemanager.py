# coding=utf-8
#https://developers.google.com/drive/v3/web/quickstart/python
from __future__ import print_function
import httplib2
import os
import io
import oauth2client
import sys
import argparse
import h5py

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from apiclient.http import MediaIoBaseDownload
from apiclient.http import MediaFileUpload

"""
DESCRIPTION
Script with class that manages operations with Google. 
Send file, uploads file and list files
"""

class GoogleManager:

    def __init__(self):

        self.SCOPES = 'https://www.googleapis.com/auth/drive'
        self.CLIENT_SECRET_FILE = 'GoogleDrive_Client_secret.json'
        self.APPLICATION_NAME = 'pythonscript'

        print("[GOOGLE MANAGER] Google Manager started")

    def init_for_upload(self,upload_file=None,upload_file_name=None):

        if upload_file and upload_file_name:
            self.upload_manager(upload_file,upload_file_name)

            print("[GOOGLE MANAGER] Will upload file")

        else:
            raise ValueError("[ERROR] Object initializer has to have file name to upload and name of uploaded file in upload mode. Initialize object with mode, upload filename and upload destination name")               

    def init_for_download(self,download_file=None):

        if download_file:                
            self.download_manager(download_file)

            print("[GOOGLE MANAGER] Will download file")

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
            credentials = tools.run_flow(flow, store)        
        return credentials

    def upload_manager(self,fileToUpload,nameToUpload):

        credentials = self.get_credentials()
        http = credentials.authorize(httplib2.Http())
        drive_service = discovery.build('drive', 'v3', http=http)
        file_metadata = {'name' : nameToUpload}
        media = MediaFileUpload(fileToUpload,resumable=True)
        file = drive_service.files().create(body=file_metadata,media_body=media,fields='id').execute()
        print(file)
        print("[GOOGLE MANAGER] File with name {} uploaded to Google Drive".format(nameToUpload))

    def download_manager(self,fileToDownload=None,list = False):   

        credentials = self.get_credentials()
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('drive', 'v3', http=http)
        results = service.files().list(pageSize=10,fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            print("[GOOGLE MANAGER] No files found.")
        else:
            for item in items:    
                name = str(item["name"].encode('ascii', 'ignore'))        
                print("[GOOGLE MANAGER] Found file -> {}".format(name))            
                if name == fileToDownload and not list:
                    credentials = self.get_credentials()
                    http = credentials.authorize(httplib2.Http())
                    drive_service = discovery.build('drive', 'v3', http=http)
                    self.download_file(item['id'],"text/plain",item['name'],drive_service)      


#drive = GoogleManager()
#drive.init_for_download("weights.h5")
#drive.init_for_download("model.json")
#drive.init_for_upload("more_images.h5","weights.h5")
#drive.init_for_upload("model_more_images.json","model.json")
#drive.init_for_list()



