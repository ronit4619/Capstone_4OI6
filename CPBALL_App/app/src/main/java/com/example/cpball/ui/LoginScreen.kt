package com.example.cpball.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.cpball.ui.components.LoginCard
import com.example.cpball.ui.theme.*
import com.example.cpball.viewmodel.AuthViewModel

@Composable
fun LoginScreen(
    onSignUpClick: () -> Unit,
    onLoginSuccess: () -> Unit,
    authViewModel: AuthViewModel = viewModel()
) {
    var username by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var loginStatus by remember { mutableStateOf<String?>(null) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(LightOrange, BackgroundGradient)))
    ) {
        BasketballBackground()

        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            LoginCard(
                username = username,
                password = password,
                onUsernameChange = { username = it },
                onPasswordChange = { password = it },
                onLoginClick = {
                    authViewModel.login(username, password) { success, message ->
                        if (success) {
                            loginStatus = "Login Successful!"
                            onLoginSuccess()
                        } else {
                            loginStatus = "Login Failed: $message"
                        }
                    }
                },
                onSignUpClick = onSignUpClick
            )
        }
    }
}
