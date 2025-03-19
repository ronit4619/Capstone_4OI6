package com.example.cpball.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import com.example.cpball.ui.theme.BackgroundGradient
import com.example.cpball.ui.theme.LightOrange
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.cpball.ui.theme.BasketballBackground

@Composable
fun LiveAnalysisScreen(onBackClick: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(LightOrange, BackgroundGradient)))
    ) {
        BasketballBackground()

        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(text = "Live Analysis 🏀", fontSize = 24.sp, fontWeight = FontWeight.Bold)

            Spacer(modifier = Modifier.height(20.dp))

            Button(
                onClick = { /* Start Camera Logic */ },
                modifier = Modifier.fillMaxWidth(0.8f),
                colors = ButtonDefaults.buttonColors(backgroundColor = Color(0xFFFF6B35))
            ) {
                Text("Start Camera", fontSize = 18.sp, color = Color.White)
            }

            Spacer(modifier = Modifier.height(20.dp))

            Button(
                onClick = onBackClick,
                modifier = Modifier.fillMaxWidth(0.8f),
                colors = ButtonDefaults.buttonColors(backgroundColor = Color.Gray)
            ) {
                Text("← Back", fontSize = 18.sp, color = Color.White)
            }
        }
    }
}
