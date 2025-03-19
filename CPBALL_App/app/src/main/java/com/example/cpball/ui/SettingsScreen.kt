package com.example.cpball.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.cpball.ui.theme.*

@Composable
fun SettingsScreen(
    onBackClick: () -> Unit,
    showBasketballBackground: Boolean,
    onToggleBasketballBackground: (Boolean) -> Unit,
    darkModeEnabled: Boolean,
    onToggleDarkMode: (Boolean) -> Unit,
    soundEffectsEnabled: Boolean,
    onToggleSoundEffects: (Boolean) -> Unit,
    enableNotifications: Boolean,
    onToggleNotifications: (Boolean) -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(LightOrange, BackgroundGradient)))
    ) {
        BasketballBackground()

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // **Title**
            Text(
                text = "Settings ⚙️",
                fontSize = 24.sp,
                fontWeight = androidx.compose.ui.text.font.FontWeight.Bold,
                color = Color.White
            )

            Spacer(modifier = Modifier.height(24.dp))

            // **Show Basketball Background Toggle**
            SettingToggle(
                text = "Show Basketball Background",
                checked = showBasketballBackground,
                onCheckedChange = onToggleBasketballBackground
            )

            // **Dark Mode Toggle**
            SettingToggle(
                text = "Dark Mode",
                checked = darkModeEnabled,
                onCheckedChange = onToggleDarkMode
            )

            // **Sound Effects Toggle**
            SettingToggle(
                text = "Enable Sound Effects",
                checked = soundEffectsEnabled,
                onCheckedChange = onToggleSoundEffects
            )

            // **Notifications Toggle**
            SettingToggle(
                text = "Enable Notifications",
                checked = enableNotifications,
                onCheckedChange = onToggleNotifications
            )

            Spacer(modifier = Modifier.height(32.dp))

            // **Back Button (Themed)**
            Button(
                onClick = onBackClick,
                colors = ButtonDefaults.buttonColors(backgroundColor = PrimaryOrange),
                shape = androidx.compose.foundation.shape.RoundedCornerShape(8.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(50.dp)
            ) {
                Text(text = "← Back", fontSize = 18.sp, color = Color.White)
            }
        }
    }
}

// **Reusable Toggle Component**
@Composable
fun SettingToggle(text: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = text,
            fontSize = 16.sp,
            color = Color.White,
            modifier = Modifier.weight(1f)
        )
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = PrimaryOrange,
                uncheckedThumbColor = Color.Gray,
                checkedTrackColor = BackgroundGradient
            )
        )
    }
}
