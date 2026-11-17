//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 11 (rank-only) — the defect of `Buggy.scala` with positional labels. COMPILES.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  */
object Case11RankOnly:

  import dimwit.*

  trait A0 derives Label
  trait A1 derives Label
  trait A2 derives Label
  trait A3 derives Label

  type Rank1 = Tensor1[A0, Float32]
  type Rank4 = Tensor4[A0, A1, A2, A3, Float32]

  // frames: [frame, height, width, channel], result: [channel]
  def perChannelMeanNhwc(frames: Rank4): Rank1 =
    frames.mean((Axis[A0], Axis[A1], Axis[A2])).relabelTo(Axis[A0])

  // decoded: [frame, channel, height, width]
  def normalise(decoded: Rank4): Rank1 =
    // NCHW and NHWC are both rank 4: nothing to reject
    perChannelMeanNhwc(decoded)
