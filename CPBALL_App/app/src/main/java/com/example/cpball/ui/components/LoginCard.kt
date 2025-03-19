package com.example.cpball.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.font.FontWeight
import com.example.cpball.ui.theme.PrimaryOrange

@Composable
fun LoginCard(
    username: String,
    password: String,
    onUsernameChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onLoginClick: () -> Unit,
    onSignUpClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth(0.85f)
            .padding(16.dp),
        shape = RoundedCornerShape(16.dp),
        elevation = 8.dp
    ) {
        Column(
            modifier = Modifier.padding(24.dp).fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(text = "Welcome Back!", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = PrimaryOrange)
            Spacer(modifier = Modifier.height(16.dp))

            InputField(value = username, onValueChange = onUsernameChange, label = "Username", modifier = Modifier.fillMaxWidth())
            Spacer(modifier = Modifier.height(12.dp))

            InputField(value = password, onValueChange = onPasswordChange, label = "Password", modifier = Modifier.fillMaxWidth())

            Spacer(modifier = Modifier.height(20.dp))

            LoginButton(onClick = onLoginClick)

            Spacer(modifier = Modifier.height(16.dp))

            SignUpPrompt(onSignUpClick = onSignUpClick)
        }
    }
}
