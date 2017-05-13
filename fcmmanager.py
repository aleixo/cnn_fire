#coding:utf-8
from pyfcm import FCMNotification
import argparse
#https://github.com/olucurious/PyFCM/blob/master/pyfcm/fcm.py
#import fcmmanager as fcm
#fcm.sendPush("alert","fire body")

"""
DESCRIPTION
Script to sent push notification through FIrebase
"""

def sendPush():
	
	data_message = {
    	"title" : "Deep Pi - Alert",
    	"body" : "System detected fire. Please check it out.",    	
	}

	push_service = FCMNotification(api_key="AAAAZW9shM0:APA91bEBRY3WT8AzjM86hWVhUaOJP9O_zKhVX8PDlVD6MVHV_7ofMu6FBDgKBt7gHCJ2RjYIOoeYkoFRPWVtlYtAF5Uou7HUe6Ky6ZLJ_I2JK1zCVjqyz3Z120IuHuVVg6W48Ywuxc29")		
	print("[FCM MANAGER] Will send push")			
	result = push_service.notify_topic_subscribers(topic_name="fire", content_available=True, data_message=data_message)

sendPush()