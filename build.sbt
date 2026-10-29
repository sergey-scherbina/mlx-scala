// mlx-scala: Apple's MLX from Scala 3, on two platforms — the JVM through FFM, and Scala Native — with one model code.
// `mlx` is the MLX API alone; `llm` (tokenizer, Qwen3, generation) depends on nothing but it; `app` reaches the world
// (model files, the console, the clock, the log) through okay-scala's effects
import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}
import scala.scalanative.build.*

ThisBuild / scalaVersion := "3.3.8"
ThisBuild / organization := "io.github.okay-effects"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Wunused:imports,privates,locals")
ThisBuild / resolvers += "okay-scala" at "https://okay-effects.github.io/okay-scala/maven"

/** native-lib/build.sh writes the libraries here: libokm (the flat layer), libmlxc, libmlx, mlx.metallib */
val nativeLib = file("native-lib/out/lib").getAbsoluteFile

val jvmNative = Seq(
  Test / fork := true,
  run / fork := true,
  javaOptions ++= Seq("--enable-native-access=ALL-UNNAMED", s"-Dokm.lib=$nativeLib/libokm.dylib", "-Xss16m"),
  run / connectInput := true)
val nativeLinking = Seq(
  nativeConfig ~= (_.withMode(Mode.releaseFast).withLTO(LTO.none)
    .withLinkingOptions(Seq(s"-L$nativeLib", "-lmlxc", s"-Wl,-rpath,$nativeLib"))))

val munit = Def.setting("org.scalameta" %%% "munit" % "1.2.3" % Test)

/** THE MLX API: arrays, ops, fast kernels, quantized matmul, safetensors — `Okm` per platform, the rest shared */
lazy val mlx = crossProject(JVMPlatform, NativePlatform).crossType(CrossType.Full).in(file("mlx"))
  .settings(name := "mlx-scala", libraryDependencies += munit.value)
  .jvmSettings(jvmNative)
  .nativeSettings(nativeLinking)

/** A LANGUAGE MODEL on the MLX API alone: a byte-level BPE tokenizer, Qwen3, its KV cache, greedy generation */
lazy val llm = crossProject(JVMPlatform, NativePlatform).crossType(CrossType.Pure).in(file("llm")).dependsOn(mlx)
  .settings(name := "mlx-scala-llm", libraryDependencies += munit.value)
  .jvmSettings(jvmNative)
  .nativeSettings(nativeLinking)

/** THE APP: generation and its measurement, the world through effects — `sbt appJVM/run` / `appNative/run` */
lazy val app = crossProject(JVMPlatform, NativePlatform).crossType(CrossType.Pure).in(file("app")).dependsOn(llm)
  .settings(name := "mlx-scala-app",
    libraryDependencies ++= Seq("io.github.okay-effects" %%% "okay-scala" % "0.1.1", munit.value))
  .jvmSettings(jvmNative)
  .nativeSettings(nativeLinking)

lazy val root = (project in file(".")).aggregate(
  (mlx.componentProjects ++ llm.componentProjects ++ app.componentProjects).map(p => p: ProjectReference): _*)
  .settings(publish / skip := true)
