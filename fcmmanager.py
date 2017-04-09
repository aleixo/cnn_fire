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
		push_service = FCMNotification(api_key="AIzaSyCAlPFgBV0Z6u9LZVWNVt-8ZTSy9mxZpA8")
		registration_id = "flafqcTKoYM:APA91bGildgQPgZQ0ss7DavoWhAHXZmMI8cK47c5eX28CAVJD_lZiJ9DLadCCMqYFKtAHCQZG6gnvCgSrDC9BLHOgW5dR_14zPy4ZedFQT2kqzGI-kX4H0wK2LY4TURRxU2WT43a037S"
		message_title = message_title
		message_body = message_body
		result = push_service.notify_single_device(registration_id=registration_id, message_title=message_title, message_body=message_body,sound="Default",badge=1)
		return result
	else:
		return "sem alarme"

