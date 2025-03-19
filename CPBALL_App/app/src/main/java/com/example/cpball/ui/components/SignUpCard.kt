package com.example.cpball.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.font.FontWeight
import com.example.cpball.ui.theme.PrimaryOrange

@Composable
fun SignUpCard(
    username: String,
    email: String,
    password: String,
    confirmPassword: String,
    onUsernameChange: (String) -> Unit,
    onEmailChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onConfirmPasswordChange: (String) -> Unit,
    onSignUpClick: () -> Unit,
    onSignInClick: () -> Unit,
    agreeToPrivacy: Boolean,
    onPrivacyChecked: (Boolean) -> Unit
) {
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
            Text(
                text = "Get Started",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = PrimaryOrange
            )
            Spacer(modifier = Modifier.height(16.dp))

            InputField(value = username, onValueChange = onUsernameChange, label = "Username", modifier = Modifier.fillMaxWidth())
            Spacer(modifier = Modifier.height(12.dp))

            InputField(value = email, onValueChange = onEmailChange, label = "Email (Optional)", modifier = Modifier.fillMaxWidth())
            Spacer(modifier = Modifier.height(12.dp))

            InputField(value = password, onValueChange = onPasswordChange, label = "Password", modifier = Modifier.fillMaxWidth())
            Spacer(modifier = Modifier.height(12.dp))

            InputField(value = confirmPassword, onValueChange = onConfirmPasswordChange, label = "Confirm Password", modifier = Modifier.fillMaxWidth())

            Spacer(modifier = Modifier.height(12.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(checked = agreeToPrivacy, onCheckedChange = onPrivacyChecked)
                Text(text = "I agree to the ")
                Text(
                    text = "Privacy Policy",
                    color = PrimaryOrange,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.clickable { /* Open Privacy Policy */ }
                )
            }

            Spacer(modifier = Modifier.height(20.dp))

            Button(
                onClick = onSignUpClick,
                enabled = agreeToPrivacy,
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(8.dp),
                colors = ButtonDefaults.buttonColors(backgroundColor = PrimaryOrange)
            ) {
                Text(text = "Create Account", fontSize = 18.sp, color = MaterialTheme.colors.onPrimary)
            }

            Spacer(modifier = Modifier.height(16.dp))


            Row {
                Text(text = "Already have an account?", fontSize = 14.sp, color = MaterialTheme.colors.onSurface)
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "Sign in here",
                    fontSize = 14.sp,
                    color = PrimaryOrange,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.clickable { onSignInClick() }
                )
            }
        }
    }
}
