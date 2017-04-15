package com.example.diogoaleixo.deeppi;

import android.util.Log;

import com.google.firebase.iid.FirebaseInstanceId;
import com.google.firebase.iid.FirebaseInstanceIdService;

/**
 * Created by diogoaleixo on 15/04/2017.
 */

public class FirebaseIdService extends FirebaseInstanceIdService {



    @Override
    public void onTokenRefresh() {
        // Get updated InstanceID token.
        final String TAG = "w";

        String refreshedToken = FirebaseInstanceId.getInstance().getToken();
        Log.w(TAG, "Refreshed token: " + refreshedToken);

        // If you want to send messages to this application instance or
        // manage this apps subscriptions on the server side, send the
        // Instance ID token to your app server.
    }
}
