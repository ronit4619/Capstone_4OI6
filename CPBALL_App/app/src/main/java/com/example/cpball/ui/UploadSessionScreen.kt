package com.example.cpball.ui

import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.cpball.ui.theme.BackgroundGradient
import com.example.cpball.ui.theme.BasketballBackground
import com.example.cpball.ui.theme.LightOrange

@Composable
fun UploadSessionScreen(
    onBackClick: () -> Unit,
    videoPickerLauncher: () -> Unit,
    selectedVideoUri: Uri?
) {
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
            Text(text = "Upload Session 📤", fontSize = 24.sp, fontWeight = FontWeight.Bold)

            Spacer(modifier = Modifier.height(20.dp))

            Box(
                modifier = Modifier
                    .fillMaxWidth(0.8f)
                    .height(120.dp)
                    .background(Color(0xFFFFA07A), RoundedCornerShape(8.dp))
                    .clickable { videoPickerLauncher() },
                contentAlignment = Alignment.Center
            ) {
                Text(text = if (selectedVideoUri == null) "Click to upload video" else "Video Selected ✅")
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
