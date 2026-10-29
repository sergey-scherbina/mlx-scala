// mlx-scala: Apple's MLX from Scala 3, on two platforms — the JVM through FFM, and Scala Native — with one model code.
// `mlx` is the MLX API alone; `llm` (tokenizer, chat templates, Qwen3, generation) depends on nothing but it
import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}
import scala.scalanative.build.*

ThisBuild / scalaVersion := "3.3.8"
ThisBuild / organization := "io.github.sergey-scherbina"
ThisBuild / version := "0.1.0"
ThisBuild / scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Wunused:imports,privates,locals")
ThisBuild / licenses := List("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.txt"))
ThisBuild / homepage := Some(url("https://github.com/sergey-scherbina/mlx-scala"))
ThisBuild / resolvers += "okay-simple" at "https://okay-effects.github.io/okay-simple/maven"
ThisBuild / developers := List(Developer("sergey-scherbina", "Sergiy Shcherbyna", "sergey.scherbina@gmail.com",
  url("https://github.com/sergey-scherbina")))
ThisBuild / scmInfo := Some(ScmInfo(url("https://github.com/sergey-scherbina/mlx-scala"),
  "scm:git:https://github.com/sergey-scherbina/mlx-scala.git"))
ThisBuild / versionScheme := Some("early-semver")

/** RELEASES, ON GITHUB PAGES: `sbt publish` writes the Maven repository into target/pages-maven, which goes to the
 * gh-pages branch under maven/ — https://sergey-scherbina.github.io/mlx-scala/maven */
val pagesRepository = settingKey[File]("the Maven repository a release is written to, for GitHub Pages")
ThisBuild / pagesRepository := (ThisBuild / baseDirectory).value / "target" / "pages-maven"
ThisBuild / publishTo := Some(MavenCache("pages", (ThisBuild / pagesRepository).value))
ThisBuild / Compile / packageDoc / publishArtifact := false

/** native-lib/build.sh writes the libraries here: libokm (the flat layer), libmlxc, libmlx, mlx.metallib */
val nativeLib = file("native-lib/out/lib").getAbsoluteFile

val jvmNative = Seq(
  Test / fork := true,
  run / fork := true,
  javaOptions ++= Seq("--enable-native-access=ALL-UNNAMED", s"-Dokm.lib=$nativeLib/libokm.dylib", "-Xss16m"))
val nativeLinking = Seq(
  nativeConfig ~= (_.withMode(Mode.releaseFast).withLTO(LTO.none).withEmbedResources(true)
    .withLinkingOptions(Seq(s"-L$nativeLib", "-lmlxc", s"-Wl,-rpath,$nativeLib"))))

val jsonValue = "0.1.1"
val munit = Def.setting("org.scalameta" %%% "munit" % "1.2.3" % Test)

/** THE MLX API: arrays, ops, fast kernels, quantized matmul, safetensors — `Okm` per platform, the rest shared */
lazy val mlx = crossProject(JVMPlatform, NativePlatform).crossType(CrossType.Full).in(file("mlx"))
  .settings(name := "mlx-scala", libraryDependencies += munit.value)
  .jvmSettings(jvmNative)
  .nativeSettings(nativeLinking)

/** A LANGUAGE MODEL on the MLX API alone: tokenizer, chat templates, Qwen3, its KV cache, generation */
lazy val llm = crossProject(JVMPlatform, NativePlatform).crossType(CrossType.Pure).in(file("llm")).dependsOn(mlx)
  .settings(name := "mlx-scala-llm",
    // JSON: the value alone, depending on nothing (okay-simple's okay-effects-json-value)
    libraryDependencies ++= Seq("io.github.okay-effects" %%% "okay-effects-json-value" % jsonValue, munit.value))
  .jvmSettings(jvmNative)
  .nativeSettings(nativeLinking)

lazy val root = (project in file(".")).aggregate(
  (mlx.componentProjects ++ llm.componentProjects).map(p => p: ProjectReference): _*)
  .settings(publish / skip := true)
