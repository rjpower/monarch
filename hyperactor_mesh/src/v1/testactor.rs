//! This module defines a test actor. It is defined in a separate module
//! (outside of [`crate::v1::testing`]) to ensure that it is compiled into
//! the bootstrap binary, which is not built in test mode (and anyway, test mode
//! does not work across crate boundaries)

use std::collections::VecDeque;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorId;
use hyperactor::Bind;
use hyperactor::Context;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor::PortRef;
use hyperactor::Unbind;
use serde::Deserialize;
use serde::Serialize;

/// A simple test actor used by various unit tests.
#[derive(Actor, Default, Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [
        GetActorId,
        Forward,
    ]
)]
pub struct TestActor;

/// A message that returns the recipient actor's id.
#[derive(Debug, Clone, Named, Bind, Unbind, Serialize, Deserialize)]
pub struct GetActorId(pub PortRef<ActorId>);

#[async_trait]
impl Handler<GetActorId> for TestActor {
    async fn handle(
        &mut self,
        cx: &Context<Self>,
        GetActorId(reply): GetActorId,
    ) -> Result<(), anyhow::Error> {
        reply.send(cx, cx.self_id().clone())?;
        Ok(())
    }
}

/// A message to forward to a visit list of ports.
/// Each port removes the next entry, and adds it to the
/// 'visited' list.
#[derive(Debug, Clone, Named, Bind, Unbind, Serialize, Deserialize)]
pub struct Forward {
    pub to_visit: VecDeque<PortRef<Forward>>,
    pub visited: Vec<PortRef<Forward>>,
}

#[async_trait]
impl Handler<Forward> for TestActor {
    async fn handle(
        &mut self,
        cx: &Context<Self>,
        Forward {
            mut to_visit,
            mut visited,
        }: Forward,
    ) -> Result<(), anyhow::Error> {
        let Some(this) = to_visit.pop_front() else {
            anyhow::bail!("unexpected forward chain termination");
        };
        visited.push(this);
        let next = to_visit.front().cloned();
        anyhow::ensure!(next.is_some(), "unexpected forward chain termination");
        next.unwrap().send(cx, Forward { to_visit, visited })?;
        Ok(())
    }
}
