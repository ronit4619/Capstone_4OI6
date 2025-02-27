package com.example.cpball.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.unit.dp
import com.example.cpball.ui.components.*
import com.example.cpball.ui.theme.*
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.font.FontWeight


@Composable
fun SignUpScreen(onSignInClick: () -> Unit) {
    var username by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var confirmPassword by remember { mutableStateOf("") }
    var agreeToPrivacy by remember { mutableStateOf(false) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(LightOrange, BackgroundGradient)))
    ) {
        // Floating basketballs
        BasketballBackground()

        // Centered Sign Up Form
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Card(
                modifier = Modifier
                    .fillMaxWidth(0.85f)
                    .padding(16.dp),
                shape = RoundedCornerShape(16.dp),
                elevation = 8.dp
            ) {
                Column(
                    modifier = Modifier
                        .padding(24.dp)
                        .fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // Title
                    Text(
                        text = "Get Started",
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = PrimaryOrange
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    // Username Field
                    InputField(
                        value = username,
                        onValueChange = { username = it },
                        label = "Username",
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    // Password Field
                    InputField(
                        value = password,
                        onValueChange = { password = it },
                        label = "Password",
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    // Confirm Password Field
                    InputField(
                        value = confirmPassword,
                        onValueChange = { confirmPassword = it },
                        label = "Confirm Password",
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    // Privacy Policy Checkbox
                    PrivacyCheckbox(
                        checked = agreeToPrivacy,
                        onCheckedChange = { agreeToPrivacy = it }
                    )

                    Spacer(modifier = Modifier.height(20.dp))

                    // Create Account Button
                    SignUpButton(
                        onClick = { /* Handle Sign Up */ },
                        isEnabled = agreeToPrivacy
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    // Sign In Link
                    SignUpPrompt(
                        onSignInClick = onSignInClick
                    )
                }
            }
        }
    }
}
