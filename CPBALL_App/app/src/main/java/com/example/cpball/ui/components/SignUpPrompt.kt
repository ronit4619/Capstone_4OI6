package com.example.cpball.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.cpball.ui.theme.PrimaryOrange

@Composable
fun SignUpPrompt(onSignUpClick: () -> Unit) {
    Row {
        Text(text = "New here?", fontSize = 14.sp)
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = "Create an account",
            fontSize = 14.sp,
            color = PrimaryOrange,
            modifier = Modifier.clickable { onSignUpClick() }
        )
    }
}
