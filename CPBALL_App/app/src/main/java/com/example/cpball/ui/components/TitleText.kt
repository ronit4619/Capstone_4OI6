package com.example.cpball.ui.components

import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

@Composable
fun TitleText() {
    Text(
        text = "Welcome Back!",
        fontSize = 24.sp,
        fontWeight = FontWeight.Bold,
        color = Color(0xFFFF4500)
    )
}
