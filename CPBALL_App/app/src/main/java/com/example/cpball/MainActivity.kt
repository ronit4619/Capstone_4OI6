package com.example.cpball

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.*
import com.example.cpball.ui.*

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            var showDashboard by remember { mutableStateOf(false) }
            var showSignUp by remember { mutableStateOf(false) }
            var showSettings by remember { mutableStateOf(false) }


            var showBasketballBackground by remember { mutableStateOf(true) }
            var darkModeEnabled by remember { mutableStateOf(false) }
            var soundEffectsEnabled by remember { mutableStateOf(true) }
            var enableNotifications by remember { mutableStateOf(true) }

            when {
                showSettings -> SettingsScreen(
                    onBackClick = { showSettings = false },
                    showBasketballBackground = showBasketballBackground,
                    onToggleBasketballBackground = { showBasketballBackground = it },
                    darkModeEnabled = darkModeEnabled,
                    onToggleDarkMode = { darkModeEnabled = it },
                    soundEffectsEnabled = soundEffectsEnabled,
                    onToggleSoundEffects = { soundEffectsEnabled = it },
                    enableNotifications = enableNotifications,
                    onToggleNotifications = { enableNotifications = it }
                )
                showDashboard -> DashboardScreen(
                    onLogoutClick = { showDashboard = false },
                    onSettingsClick = { showSettings = true }
                )
                showSignUp -> SignUpScreen(
                    onSignInClick = { showSignUp = false },
                    onSignUpClick = { showSignUp = false; showDashboard = true }
                )
                else -> LoginScreen(
                    onSignUpClick = { showSignUp = true },
                    onLoginSuccess = { showDashboard = true }
                )
            }
        }
    }
}
