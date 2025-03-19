package com.example.cpball.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import com.example.cpball.ui.components.SignUpCard
import com.example.cpball.ui.theme.*

@Composable
fun SignUpScreen(
    onSignInClick: () -> Unit,
    onSignUpClick: () -> Unit
) {
    var username by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var confirmPassword by remember { mutableStateOf("") }
    var agreeToPrivacy by remember { mutableStateOf(false) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(LightOrange, BackgroundGradient)))
    ) {
        BasketballBackground()

        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            SignUpCard(
                username = username,
                email = email,
                password = password,
                confirmPassword = confirmPassword,
                onUsernameChange = { username = it },
                onEmailChange = { email = it },
                onPasswordChange = { password = it },
                onConfirmPasswordChange = { confirmPassword = it },
                onSignUpClick = onSignUpClick,
                onSignInClick = onSignInClick,
                agreeToPrivacy = agreeToPrivacy,
                onPrivacyChecked = { agreeToPrivacy = it }
            )
        }
    }
}
