import redis.clients.jedis.{Jedis, JedisPool, JedisPoolConfig}

object JedisConnectionPool {

  private val config = new JedisPoolConfig()
//  config.setMaxTotal(20)
//  config.setMaxIdle(10)
//  config.setTestOnBorrow(true)

  private val pool = new JedisPool(config, "10.205.48.52", 6379)

  def getConnection: Jedis = {
    pool.getResource
  }

  def close(): Unit = {
    pool.close()
  }

}
