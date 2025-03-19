package com.example.cpball.ui

import MainMenu
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.material.icons.filled.Settings
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import com.example.cpball.ui.theme.BackgroundGradient
import com.example.cpball.ui.theme.LightOrange
import com.example.cpball.ui.theme.BasketballBackground
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun DashboardScreen(
    onLogoutClick: () -> Unit,
    onSettingsClick: () -> Unit
) {
    var currentPage by remember { mutableStateOf("menu") }
    var selectedVideoUri by remember { mutableStateOf<Uri?>(null) }

    val videoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri: Uri? -> selectedVideoUri = uri }
    )

    Scaffold(
        topBar = { /* Removed TopAppBar & Title */ }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Brush.verticalGradient(listOf(LightOrange, BackgroundGradient)))
                .padding(paddingValues)
        ) {
            BasketballBackground()

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(top = 50.dp), // Keeps spacing at the top
                horizontalAlignment = Alignment.End, // Aligns settings icon to right
                verticalArrangement = Arrangement.Top
            ) {
                // ⚙️ Settings Icon - Positioned at the top-right
                IconButton(
                    onClick = onSettingsClick,
                    modifier = Modifier
                        .padding(end = 16.dp)
                        .size(40.dp)
                ) {
                    Icon(
                        imageVector = androidx.compose.material.icons.Icons.Default.Settings,
                        contentDescription = "Settings",
                        tint = Color.White
                    )
                }
            }

            // **Center Menu**
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center // ✅ Center menu vertically
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    when (currentPage) {
                        "menu" -> MainMenu(
                            onLiveClick = { currentPage = "live" },
                            onUploadClick = { currentPage = "upload" },
                            onLogoutClick = onLogoutClick
                        )
                        "live" -> LiveAnalysisScreen(onBackClick = { currentPage = "menu" })
                        "upload" -> UploadSessionScreen(
                            onBackClick = { currentPage = "menu" },
                            videoPickerLauncher = { videoPickerLauncher.launch("video/*") },
                            selectedVideoUri = selectedVideoUri
                        )
                    }
                }
            }
        }
    }
}




