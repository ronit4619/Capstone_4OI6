package com.example.cpball.ui.theme

import androidx.compose.foundation.layout.*
import androidx.compose.material.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.random.Random

@Composable
fun BasketballBackground() {
    val usedPositions = mutableListOf<Pair<Int, Int>>() // Keeps track of used positions
    val basketballs = mutableListOf<Pair<Int, Int>>() // Stores final non-overlapping positions

    val screenWidth = 350  // Approximate screen width in dp
    val screenHeight = 800 // Approximate screen height in dp
    val minDistance = 100  // Minimum distance between basketballs to avoid overlap

    repeat(10) {  // Generate 10 basketballs
        var x: Int
        var y: Int
        var validPosition: Boolean

        do {
            x = Random.nextInt(20, screenWidth)
            y = Random.nextInt(50, screenHeight)

            // Check if the position is far enough from existing basketballs
            validPosition = usedPositions.none { (prevX, prevY) ->
                (kotlin.math.abs(x - prevX) < minDistance) &&
                        (kotlin.math.abs(y - prevY) < minDistance)
            }

        } while (!validPosition)  // Keep generating until a non-overlapping position is found

        usedPositions.add(Pair(x, y))  // Save the position
        basketballs.add(Pair(x, y))
    }

    Box(modifier = Modifier.fillMaxSize()) {
        basketballs.forEach { (x, y) ->
            Text(
                text = "🏀",
                fontSize = Random.nextInt(30, 50).sp,  // Random sizes
                modifier = Modifier.absoluteOffset(x = x.dp, y = y.dp)
            )
        }
    }
}
