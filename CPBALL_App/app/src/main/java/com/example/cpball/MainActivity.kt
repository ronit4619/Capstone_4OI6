package com.example.cpball

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.*
import com.example.cpball.ui.LoginScreen
import com.example.cpball.ui.SignUpScreen
import com.example.cpball.ui.theme.CPBallTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CPBallTheme {
                var showLoginScreen by remember { mutableStateOf(true) }

                if (showLoginScreen) {
                    LoginScreen(onSignUpClick = { showLoginScreen = false })
                } else {
                    SignUpScreen(onSignInClick = { showLoginScreen = true })
                }
            }
        }
    }
}
