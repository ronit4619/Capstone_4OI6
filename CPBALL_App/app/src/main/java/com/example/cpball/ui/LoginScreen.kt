package com.example.cpball.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.cpball.ui.components.LoginCard
import com.example.cpball.ui.theme.*
import com.example.cpball.viewmodel.AuthViewModel
import com.example.cpball.ui.theme.BasketballBackground


@Composable
fun LoginScreen(onSignUpClick: () -> Unit, authViewModel: AuthViewModel = viewModel()) {
    var username by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var loginStatus by remember { mutableStateOf<String?>(null) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(LightOrange, BackgroundGradient)))
    ) {
        // Floating basketballs
        BasketballBackground()

        // Centered Login Form
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            LoginCard(
                username = username,
                password = password,
                onUsernameChange = { username = it },
                onPasswordChange = { password = it },
                onLoginClick = {
                    authViewModel.login(username, password) { success, message ->
                        loginStatus = if (success) "Login Successful!" else "Login Failed: $message"
                    }
                },
                onSignUpClick = onSignUpClick
            )
        }
    }
}


