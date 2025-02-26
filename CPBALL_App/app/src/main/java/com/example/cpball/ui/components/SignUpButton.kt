package com.example.cpball.ui.components

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.cpball.ui.theme.PrimaryOrange
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height


@Composable
fun SignUpButton(onClick: () -> Unit, isEnabled: Boolean) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(backgroundColor = PrimaryOrange),
        shape = RoundedCornerShape(8.dp),
        modifier = Modifier
            .fillMaxWidth()
            .height(50.dp),
        enabled = isEnabled
    ) {
        Text(text = "Create Account", fontSize = 18.sp, color = Color.White)
    }
}
