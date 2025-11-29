// build.gradle.kts
// [构建] Gradle Kotlin DSL：声明 grpc-stub、resilience4j、spring-boot-starter 等依赖
// 说明：用于构建 Gateway (Kotlin + Spring Boot) 项目。

import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
	kotlin("jvm") version "1.9.24"
	kotlin("plugin.spring") version "1.9.24"
	id("org.springframework.boot") version "3.3.2"
	id("io.spring.dependency-management") version "1.1.5"
	id("com.google.protobuf") version "0.9.4"
}

group = "com.genshin.ai"
version = "0.0.1-SNAPSHOT"

java.sourceCompatibility = JavaVersion.VERSION_21

repositories {
    maven { url = uri("https://maven.aliyun.com/repository/public") }
    maven { url = uri("https://maven.aliyun.com/repository/google") }
    maven { url = uri("https://maven.aliyun.com/repository/gradle-plugin") }
    mavenCentral()
}

dependencies {
	implementation("org.springframework.boot:spring-boot-starter-webflux")
	implementation("org.springframework.boot:spring-boot-starter-validation")
	implementation("org.springframework.boot:spring-boot-starter-actuator")
	implementation("com.fasterxml.jackson.module:jackson-module-kotlin")
	implementation("io.grpc:grpc-netty-shaded:1.65.1")
	implementation("io.grpc:grpc-stub:1.65.1")
	implementation("io.grpc:grpc-protobuf:1.65.1")
	implementation("com.google.protobuf:protobuf-java:3.25.3")
	implementation("io.github.resilience4j:resilience4j-spring-boot3:2.2.0")
	implementation("io.github.resilience4j:resilience4j-kotlin:2.2.0")
	implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
	implementation("org.jetbrains.kotlinx:kotlinx-coroutines-reactor:1.8.1")
	implementation("io.projectreactor.kotlin:reactor-kotlin-extensions:1.2.3")
	implementation("javax.annotation:javax.annotation-api:1.3.2")
	compileOnly("jakarta.annotation:jakarta.annotation-api:2.1.1")

	testImplementation("org.springframework.boot:spring-boot-starter-test")
	testImplementation("org.mockito.kotlin:mockito-kotlin:5.3.1")
	testImplementation("io.projectreactor:reactor-test")
	testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
}

protobuf {
	protoc {
		artifact = "com.google.protobuf:protoc:3.25.3"
	}
    plugins {
        create("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:1.65.1"
        }
    }
    generateProtoTasks {
        all().forEach { task ->
            task.plugins {
                create("grpc")
            }
        }
    }
}

sourceSets {
	main {
		proto {
			srcDir("../proto")
		}
		java {
			srcDir("build/generated/source/proto/main/java")
			srcDir("build/generated/source/proto/main/grpc")
		}
	}
}

kotlin {
	jvmToolchain(21)
}

tasks.withType<KotlinCompile> {
	kotlinOptions {
		freeCompilerArgs = freeCompilerArgs + "-Xjsr305=strict"
		jvmTarget = "21"
	}
}

tasks.withType<Test> {
	useJUnitPlatform()
}
