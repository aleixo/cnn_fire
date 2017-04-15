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

def sendPush(message_title,message_body):
	
	if message_body is not "":
		push_service = FCMNotification(api_key="AAAAZW9shM0:APA91bEBRY3WT8AzjM86hWVhUaOJP9O_zKhVX8PDlVD6MVHV_7ofMu6FBDgKBt7gHCJ2RjYIOoeYkoFRPWVtlYtAF5Uou7HUe6Ky6ZLJ_I2JK1zCVjqyz3Z120IuHuVVg6W48Ywuxc29")
		registration_id = "czegMzrA-sQ:APA91bHJIoqQlJNFoVXsLhfN1DpfxgriWQZ4tt0JlR-I1sUZrbasEb7u-VefBiFOgTRNiE0XvseWTZPsCOyHv2LdK452KZm4EN0q73nqgr1hlSwiF1OSRriYG-ueO03i7fpKz27gFGpq"
		message_title = message_title
		message_body = message_body
		print("[FCM MANAGER] Will send push")
		result = push_service.notify_single_device(registration_id=registration_id, message_title=message_title, message_body=message_body,sound="Default",badge=1)
		return result
	else:
		return "sem alarme"

sendPush("title","body")