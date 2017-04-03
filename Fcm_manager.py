#coding:utf-8
from pyfcm import FCMNotification
import argparse
#https://github.com/olucurious/PyFCM/blob/master/pyfcm/fcm.py
#import firebase_server as fcm
#fcm.sendPush("","")

"""
DESCRIPTION
Script to sent push notification through FIrebase
"""

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

