package com.example.diogoaleixo.deeppi;
import android.app.NotificationManager;
import android.content.Context;
import android.content.Intent;
import android.support.v4.app.NotificationCompat;
import android.util.Log;
import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

/**
 * Created by diogoaleixo on 15/04/2017.
 */

public class FirebaseNotificationService extends FirebaseMessagingService {

    public void onMessageReceived(RemoteMessage remoteMessage) {
        // ...
        String TAG = "INFO";
        // TODO(developer): Handle FCM messages here.
        // Not getting messages here? See why this may be: https://goo.gl/39bRNJ
        Log.w(TAG, "From: " + remoteMessage.getFrom());

        // Check if message contains a data payload.
        if (remoteMessage.getData().size() > 0) {
            Log.w(TAG, "Message data payload: " + remoteMessage.getData());
        }

        // Check if message contains a notification payload.
        if (remoteMessage.getNotification() != null) {
            Log.w(TAG, "Message Notification Body: " + remoteMessage.getNotification().getBody());
        }

        // Also if you intend on generating your own notifications as a result of a received FCM
        // message, here is where that should be initiated. See sendNotification method below.
        NotificationCompat.Builder nBuilder = new NotificationCompat.Builder(this)
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentTitle(remoteMessage.getNotification().getBody())
                .setContentText(remoteMessage.getData().toString());

        int nId = 001;
        NotificationManager mMgr = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        mMgr.notify(nId,nBuilder.build());
    }
}
